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
    Fixed-batch carry for RTify-style halting.

    - inner_carry  : TRM latent state (Z_S / Z_R)
    - steps        : per-sample step counter            [B] int32
    - halted       : per-sample halted mask             [B] bool
    - last_logits  : frozen logits for halted samples   [B, L, V] float32
    - phi          : accumulated evidence Φ             [B] float32
    """
    inner_carry:  TRMCarry
    steps:        torch.Tensor
    halted:       torch.Tensor
    last_logits:  torch.Tensor
    phi:          torch.Tensor


class TRMRtifyConfig(TRMConfig):
    """
    TRMConfig extended with RTify halting fields.

    theta_init is derived from the expected Softplus(0) = log(2) ≈ 0.693
    value emitted by fw at random initialisation.  Setting theta_init to
    (2/3 * halt_max_steps * log(2)) makes the model halt at roughly the
    2/3 point initially, giving the task loss time to stabilise before
    the halting signal becomes active.
    """
    model_config = ConfigDict(extra="allow")

    halt_max_steps: int = 16

    # θ ≈ (2/3 * halt_max_steps * log(2)) at random init → halts at ~2/3 of steps
    theta_init: float = 7.4          # for halt_max_steps=16: (2/3)*16*ln(2) ≈ 7.4
    train_theta: bool = False        # if False, theta is a frozen buffer
    # NOTE: if train_theta=True, use a separate, smaller lr for theta
    # (≈ 0.1 × main lr), because ∂τ/∂θ ≈ 1/log(2) ≈ 1.44 at init,
    # so theta moves fast relative to other parameters.

    detach_fw_input: bool = True     # fw reads z.detach() — prevents halt
                                     # gradients from reshaping Z_S
    train_fixed_steps: bool = True   # DDP-safe: disable threshold halting
                                     # during training; only halt at max steps

    fw_hidden_mult: float = 1.0      # hidden-size multiplier inside fw MLP


class TRM_Rtify(nn.Module):
    """
    TRM + RTify-style monotone evidence halting.

    One call to forward() = ONE reasoning step (supervision episode).

    Evidence network fw:
        e_m = Softplus(fw(z_summary_m))   > 0  always
        Φ_m = Φ_{m-1} + e_m              strictly increasing

    Halt condition (inference / train_fixed_steps=False):
        halt when Φ_m >= θ  OR  steps >= halt_max_steps

    Training loss (added by RTifyLossHead, not here):
        L_m = CE(logits_m, y) + λ * e_m

    Penalising e_m at every step implicitly penalises the stopping time
    because Φ is the cumulative sum of evidence — no survival function,
    no Taylor approximation, no RL required.

    z_summary is the MEAN of Z_S over the sequence dimension.  Using the
    mean rather than a single token makes the readiness signal robust to
    positional accidents and gives fw a richer input.
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

        # Evidence network: Z_S mean -> positive scalar evidence e > 0
        # g is the pre-Softplus logit; e = Softplus(g)
        self.fw_fc1 = CastedLinear(D, H, bias=True)
        self.fw_fc2 = CastedLinear(H, 1, bias=True)

        # Threshold θ
        theta = torch.tensor(float(self.config.theta_init), dtype=torch.float32)
        if self.config.train_theta:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta, persistent=True)


    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRM_RtifyCarry:
        B      = batch["inputs"].shape[0]
        device = batch["inputs"].device

        inner_carry = self.inner.init_carry(B, device)

        last_logits = torch.zeros(
            (B, self.config.seq_len, self.config.vocab_size),
            device=device,
            dtype=torch.float32,
        )

        return TRM_RtifyCarry(
            inner_carry = inner_carry,
            steps       = torch.zeros((B,), device=device, dtype=torch.int32),
            halted      = torch.zeros((B,), device=device, dtype=torch.bool),
            last_logits = last_logits,
            phi         = torch.zeros((B,), device=device, dtype=torch.float32),
        )


    def _fw(self, z_summary: torch.Tensor) -> torch.Tensor:
        """
        Map a [B, D] hidden summary to per-sample positive evidence e ∈ (0, ∞).

        Returns:
            e : [B]  Softplus output, strictly positive
        """
        h = F.relu(self.fw_fc1(z_summary))      # [B, H]
        g = self.fw_fc2(h).squeeze(-1)           # [B]
        e = F.softplus(g)                        # [B]  > 0
        return e


    def forward(
        self,
        carry: TRM_RtifyCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRM_RtifyCarry, Dict[str, torch.Tensor]]:

        inputs = batch["inputs"]

        new_inner_carry, logits = self.inner(inputs, carry.inner_carry)
        logits = logits.to(torch.float32)        # [B, L, V]

        z_summary = new_inner_carry.Z_S.mean(dim=1)   # [B, D]
        if self.config.detach_fw_input:
            z_summary = z_summary.detach()

        e = self._fw(z_summary)                  # [B]  > 0, in compute graph

        with torch.no_grad():
            prev_halted = carry.halted           # [B] bool

            active  = ~prev_halted               # [B] bool
            steps   = carry.steps + active.to(torch.int32)   # [B]

            phi = carry.phi + torch.where(
                active,
                e.to(torch.float32),
                torch.zeros_like(e),
            )                                    # [B]

            halted = steps >= int(self.config.halt_max_steps)

            allow_threshold = (not self.training) or (not self.config.train_fixed_steps)
            if allow_threshold:
                halted = halted | (phi >= self.theta)

            halted     = prev_halted | halted    # [B]
            just_halted = (~prev_halted) & halted  # [B]  first time halting

        freeze_mask = prev_halted.view(-1, 1, 1)   # [B, 1, 1]

        frozen_logits = torch.where(freeze_mask, carry.last_logits, logits)
        ZS = torch.where(freeze_mask, carry.inner_carry.Z_S, new_inner_carry.Z_S)
        ZR = torch.where(freeze_mask, carry.inner_carry.Z_R, new_inner_carry.Z_R)

        new_carry = TRM_RtifyCarry(
            inner_carry = TRMCarry(Z_S=ZS, Z_R=ZR),
            steps       = steps,
            halted      = halted,
            last_logits = frozen_logits,
            phi         = phi,
        )

        outputs = {
            "logits":      frozen_logits,   # [B, L, V]
            "evidence":    e,               # [B]  > 0, in compute graph
            "active":      active,          # [B]  bool — which samples contributed
            "phi":         phi,             # [B]  accumulated Φ (detached)
            "theta":       self.theta,      # scalar tensor
            "just_halted": just_halted,     # [B]  bool
        }
        return new_carry, outputs