from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, ValidationError

from models.recursive_reasoning.TRM_NoACT import TRMCarry, TRMConfig, TRM
from models.layers import CastedLinear

IGNORE_LABEL_ID = -100


@dataclass
class TRM_RtifyCarry:
    """
    Fixed-batch carry, ACT-like.

    - inner_carry: TRM latent state (Z_S/Z_R)
    - steps: per-sample step counter
    - halted: per-sample halted mask
    - last_logits: frozen logits for halted samples
    - phi: accumulated evidence Φ
    """
    inner_carry: TRMCarry
    steps: torch.Tensor        # [B] int32
    halted: torch.Tensor       # [B] bool
    last_logits: torch.Tensor  # [B,L,V] float32
    phi: torch.Tensor          # [B] float32


class TRMRtifyConfig(TRMConfig):
    """
    TRMConfig + Rtify halting fields.

    We keep the interface similar to ACT:
      - halt_max_steps: max steps allowed
      - theta_init: initial threshold
      - detach_fw_input: whether fw sees detached z (recommended True)
      - train_fixed_steps: if True, do NOT early-halt during training (DDP-safe)
    """
    model_config = ConfigDict(extra="allow")

    halt_max_steps: int = 16

    theta_init: float = 8.0
    train_theta: bool = False          # if False, theta is frozen
    theta_reg: float = 0.0             # optional: encourage smaller theta if train_theta

    detach_fw_input: bool = True       # fw reads z.detach()
    train_fixed_steps: bool = True     # DDP-safe: only halt by max steps during training

    # evidence net shape
    fw_hidden_mult: float = 1.0        # hidden size multiplier inside fw MLP


class TRM_Rtify(nn.Module):
    """
    TRM + monotone evidence halting wrapper (RTiFy-style, hard threshold).

    One call = ONE reasoning step.

    Training:
      - usually run fixed number of steps (train_fixed_steps=True) to keep DDP aligned
      - loss head trains fw via readiness BCE + evidence penalty

    Inference/Eval:
      - can halt early when Φ > θ (if you set train_fixed_steps=False or run in eval mode)
    """

    def __init__(self, cfg_dict: Dict):
        super().__init__()
        try:
            self.config = TRMRtifyConfig(**cfg_dict)
        except ValidationError as e:
            raise ValueError(f"[TRM_Rtify] invalid config: {e}") from e

        self.inner = TRM(cfg_dict)

        D = self.config.hidden_size
        H = int(self.config.fw_hidden_mult * D)

        # Evidence network outputs a logit g; evidence is Softplus(g) > 0
        self.fw_fc1 = CastedLinear(D, H, bias=True)
        self.fw_fc2 = CastedLinear(H, 1, bias=True)

        # Threshold θ
        theta = torch.tensor(float(self.config.theta_init), dtype=torch.float32)
        if self.config.train_theta:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta, persistent=True)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRM_RtifyCarry:
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device

        inner_carry = self.inner.init_carry(B, device)

        last_logits = torch.zeros(
            (B, self.config.seq_len, self.config.vocab_size),
            device=device,
            dtype=torch.float32,
        )

        return TRM_RtifyCarry(
            inner_carry=inner_carry,
            steps=torch.zeros((B,), device=device, dtype=torch.int32),
            halted=torch.zeros((B,), device=device, dtype=torch.bool),
            last_logits=last_logits,
            phi=torch.zeros((B,), device=device, dtype=torch.float32),
        )

    def _fw(self, z_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          g: [B] logit (for readiness BCE)
          e: [B] positive evidence (Softplus)
        """
        h = F.relu(self.fw_fc1(z_summary))
        g = self.fw_fc2(h).squeeze(-1)          # [B]
        e = F.softplus(g)                       # [B] > 0
        return g, e

    def forward(self, carry: TRM_RtifyCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_RtifyCarry, Dict[str, torch.Tensor]]:
        inputs = batch["inputs"]

        new_inner_carry, logits = self.inner(inputs, carry.inner_carry)
        logits = logits.to(torch.float32)  # [B,L,V]

        # evidence from Z_S summary token 0
        z_summary = new_inner_carry.Z_S[:, 0]  # [B,D]
        if self.config.detach_fw_input:
            z_summary = z_summary.detach()

        g, e = self._fw(z_summary)  # g:[B], e:[B]

        with torch.no_grad():
            prev_halted = carry.halted
            steps = torch.where(prev_halted, carry.steps, carry.steps + 1)

            phi = torch.where(prev_halted, carry.phi, carry.phi + e.to(torch.float32))

            halted = steps >= int(self.config.halt_max_steps)
            allow_threshold = (not self.training) or (not self.config.train_fixed_steps)
            if allow_threshold:
                halted = halted | (phi >= self.theta)

            halted = prev_halted | halted
            just_halted = (~prev_halted) & halted

        
        freeze_mask = prev_halted.view(-1, 1, 1)  # [B,1,1]

        frozen_logits = torch.where(freeze_mask, carry.last_logits, logits)
        ZS = torch.where(freeze_mask, carry.inner_carry.Z_S, new_inner_carry.Z_S)
        ZR = torch.where(freeze_mask, carry.inner_carry.Z_R, new_inner_carry.Z_R)

        new_carry = TRM_RtifyCarry(
            inner_carry=TRMCarry(Z_S=ZS, Z_R=ZR),
            steps=steps,
            halted=halted,
            last_logits=frozen_logits,
            phi=phi,
        )

        outputs = {
            "logits": frozen_logits,     # [B,L,V]
            "g_logit": g,                # [B]
            "evidence": e,               # [B] > 0
            "phi": phi,                  # [B]
            "theta": self.theta,         # scalar tensor
            "just_halted": just_halted,  # [B]
        }
        return new_carry, outputs