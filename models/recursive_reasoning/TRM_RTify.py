from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

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
    model_config = ConfigDict(extra="allow")

    halt_max_steps:    int   = 16
    halt_warmup_steps: int   = 1000

    theta_init:        float = 7.4
    theta_min:         float = 3.0   # hard floor — theta = theta_min + Softplus(psi)
    train_theta:       bool  = False

    detach_fw_input:   bool  = True
    train_fixed_steps: bool  = True

    fw_hidden_mult:    float = 1.0
    lambda_halt:       float = 1e-3
    lambda_ready:      float = 0.1
    lambda_tau:        float = 0.0   # set > 0 to enable tau loss and train theta


class TRM_Rtify(nn.Module):
    """
    TRM + RTify-style monotone evidence halting.

    Evidence network fw:
        e_m = Softplus(fw(z_summary_m))   > 0  always
        Phi_m = Phi_{m-1} + e_m           strictly increasing

    Halt condition:
        halt when Phi_m >= theta  OR  steps >= halt_max_steps

    Theta reparametrisation:
        theta = theta_min + Softplus(psi)

    psi is the learned parameter. theta is always strictly above theta_min
    by construction — no clamping or floor penalty needed.

    The tau loss pushes psi toward -inf (theta toward theta_min).
    The task loss pushes back when halting too early hurts accuracy.
    The fixed point is where these forces balance, always at theta > theta_min.

    Differentiable stopping time (RTify Taylor approximation):
        tau = t* - (Phi_{t*} - theta) / e_{t*}

        d(tau)/d(theta) = 1 / e_{t*}  > 0
        d(tau)/d(psi)   = d(tau)/d(theta) * d(theta)/d(psi)
                        = sigmoid(psi) / e_{t*}  > 0

    RTifyLossHead applies:
        L_tau = lambda_tau * mean(tau[just_halted])

    giving psi (and therefore theta) a gradient:
        dL_tau/d(psi) = -lambda_tau * sigmoid(psi) / e_{t*}
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

        self.fw_fc1 = CastedLinear(D, H, bias=True)
        self.fw_fc2 = CastedLinear(H, 1, bias=True)

        # ── Theta initialisation ───────────────────────────────────────
        # theta = theta_min + Softplus(psi)
        # Solve for psi_init given theta_init and theta_min:
        #   Softplus(psi_init) = theta_init - theta_min
        #   psi_init = log(exp(theta_init - theta_min) - 1)
        theta_min  = float(self.config.theta_min)
        theta_init = float(self.config.theta_init)
        assert theta_init > theta_min, (
            f"theta_init ({theta_init}) must be > theta_min ({theta_min})"
        )
        gap      = theta_init - theta_min                          # > 0
        psi_init = math.log(math.exp(gap) - 1.0)                  # softplus inverse

        if self.config.train_theta:
            self.psi = nn.Parameter(
                torch.tensor(psi_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "psi", torch.tensor(psi_init, dtype=torch.float32), persistent=True
            )

    @property
    def theta(self) -> torch.Tensor:
        """
        theta = theta_min + Softplus(psi)

        Always strictly above theta_min. Differentiable w.r.t. psi.
        d(theta)/d(psi) = sigmoid(psi) in (0, 1).
        """
        return self.config.theta_min + F.softplus(self.psi)

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

    def _fw(self, z_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [B, D] -> g [B], e [B]
        g: pre-Softplus logit  (used for readiness BCE)
        e: Softplus(g) > 0     (used for phi accumulation, halt penalty, tau)
        """
        h = F.relu(self.fw_fc1(z_summary))   # [B, H]
        g = self.fw_fc2(h).squeeze(-1)        # [B]
        e = F.softplus(g)                     # [B]  > 0
        return g, e

    def forward(
        self,
        carry: TRM_RtifyCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TRM_RtifyCarry, Dict[str, torch.Tensor]]:

        inputs = batch["inputs"]

        new_inner_carry, logits = self.inner(inputs, carry.inner_carry)
        logits = logits.to(torch.float32)

        z_summary = new_inner_carry.Z_S.mean(dim=1)   # [B, D]
        if self.config.detach_fw_input:
            z_summary = z_summary.detach()

        g, e = self._fw(z_summary)            # [B], [B] — e in compute graph

        # Materialise theta once — keeps the graph clean and avoids
        # recomputing Softplus(psi) multiple times per forward pass.
        theta = self.theta                    # scalar, in graph if train_theta=True

        # ── Halting logic (no grad) ────────────────────────────────────
        with torch.no_grad():
            prev_halted = carry.halted
            active      = ~prev_halted
            steps       = carry.steps + active.to(torch.int32)

            phi = carry.phi + torch.where(
                active,
                e.to(torch.float32),
                torch.zeros_like(e),
            )

            halted = steps >= int(self.config.halt_max_steps)

            force_fixed     = batch.get("force_fixed_steps", False)
            allow_threshold = (
                (not self.training or not self.config.train_fixed_steps)
                and not force_fixed
            )
            if allow_threshold:
                halted = halted | (phi >= theta)

            halted      = prev_halted | halted
            just_halted = (~prev_halted) & halted   # [B]  first halt this step

        # ── Differentiable stopping time τ ────────────────────────────
        # Outside no_grad: e and theta (via psi) are live in the graph.
        # phi and steps are detached — constants in this expression.
        # Only computed when threshold halting is active so psi never
        # receives spurious gradients from max-step halts during warmup.
        if allow_threshold:
            tau = steps.float() - (phi - theta) / e.clamp_min(1e-6)   # [B]
        else:
            tau = torch.zeros_like(e)   # no graph connection to psi/theta

        # ── Freeze halted samples ──────────────────────────────────────
        freeze_mask   = prev_halted.view(-1, 1, 1)
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
            "logits":          frozen_logits,
            "g_logit":         g,               # [B]  pre-Softplus, for readiness BCE
            "evidence":        e,               # [B]  > 0, in compute graph
            "active":          active,          # [B]  bool
            "phi":             phi,             # [B]  detached
            "theta":           theta,           # scalar, in graph if train_theta=True
            "just_halted":     just_halted,     # [B]  bool
            "tau":             tau,             # [B]  in graph via e and psi/theta
            "allow_threshold": allow_threshold, # bool — gates tau loss in loss head
        }
        return new_carry, outputs