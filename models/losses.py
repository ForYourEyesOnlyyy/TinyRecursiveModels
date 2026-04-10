from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100



def s(x, epsilon=1e-30):
    return torch.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1,
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs,
        index=transformed_labels.to(torch.long).unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)



class ACTLossHead(nn.Module):
    """Fixed-batch ACT loss head (Q-learning style, from HRM)."""

    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy"):
        super().__init__()
        self.model   = model
        self.loss_fn = globals()[loss_type]

    def forward(
        self,
        *,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(carry=carry, batch=batch)
        labels = batch["labels"]

        with torch.no_grad():
            preds         = torch.argmax(outputs["logits"], dim=-1)
            mask          = (labels != IGNORE_LABEL_ID)
            loss_counts   = mask.sum(-1)
            loss_divisor  = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct    = mask & (preds == labels)
            seq_is_correct = (is_correct.sum(-1) == loss_counts)
            just_halted   = outputs.get("just_halted", new_carry.halted)
            valid_metrics = just_halted & (loss_counts > 0)

            metrics = {
                "count":           valid_metrics.sum(),
                "accuracy":        torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0.0).sum(),
                "exact_accuracy":  (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":           torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        lm_loss = (self.loss_fn(
            outputs["logits"],
            labels,
            ignore_index=IGNORE_LABEL_ID,
            valid_mask=mask,
        ) / loss_divisor).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        q_continue_loss = torch.tensor(0.0, device=lm_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

        metrics.update({
            "lm_loss":         lm_loss.detach(),
            "q_halt_loss":     q_halt_loss.detach(),
            "q_continue_loss": q_continue_loss.detach(),
        })

        total_loss      = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs} if return_keys else None
        all_finish       = new_carry.halted.all()
        return new_carry, total_loss, metrics, detached_outputs, all_finish


class RTifyLossHead(nn.Module):
    """
    Per-step loss head for TRM_Rtify.

    Loss at supervision step m:

        L_m = CE(logits_m, y)
            + lambda_halt  * mean(e_m[active])
            + lambda_ready * BCE(g_m, frac_correct[active])
            + lambda_tau   * mean(tau[just_halted])

    where:
        active      = ~prev_halted
        tau         = t* - (Phi_{t*} - theta) / e_{t*}   (RTify Taylor approx)

    lambda_tau=0 disables the tau term entirely (default).
    Set lambda_tau > 0 together with train_theta=True to make theta trainable.
    """

    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy"):
        super().__init__()
        self.model        = model
        self.loss_fn      = globals()[loss_type]
        self.lambda_halt  = float(model.config.lambda_halt)
        self.lambda_ready = float(model.config.lambda_ready)
        self.lambda_tau   = float(model.config.lambda_tau)

    def forward(
        self,
        *,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        new_carry, outputs = self.model(carry=carry, batch=batch)

        labels   = batch["labels"]
        logits   = outputs["logits"]        # [B, L, V]
        g_logit  = outputs["g_logit"]       # [B]  pre-Softplus, in compute graph
        evidence = outputs["evidence"]      # [B]  > 0, in compute graph
        active   = outputs["active"]        # [B]  bool: ~prev_halted
        phi      = outputs["phi"]           # [B]  detached Phi
        theta    = outputs["theta"]         # scalar, in graph if train_theta=True
        tau      = outputs["tau"]           # [B]  in graph via e and theta
        just_halted = outputs["just_halted"]  # [B]  bool
        allow_threshold = outputs["allow_threshold"]

        mask         = (labels != IGNORE_LABEL_ID)             # [B, L]
        loss_counts  = mask.sum(-1)                            # [B]
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # [B, 1]

        # ── Task loss ──────────────────────────────────────────────────
        lm_loss = (
            self.loss_fn(
                logits,
                labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=mask,
            ) / loss_divisor
        ).sum()

        # ── Correctness targets (no grad) ──────────────────────────────
        with torch.no_grad():
            preds       = logits.argmax(-1)                                # [B, L]
            is_correct  = mask & (preds == labels)                         # [B, L]
            # Soft target: fraction of tokens correct in [0, 1].
            # Better than binary for hard tasks — fw gets a gradient even
            # when the model is only partially correct.
            seq_correct = is_correct.float().sum(-1) / loss_counts.float().clamp_min(1) # [B]

        # ── Readiness loss ─────────────────────────────────────────────
        # fw predicts how correct the current answer is.
        # Weighted to active samples only — halted samples have frozen Z_S.
        n_active = active.float().sum().clamp_min(1.0)
        readiness_loss = F.binary_cross_entropy_with_logits(
            g_logit,
            seq_correct,
            weight=active.float(),
            reduction="sum",
        ) / n_active
        readiness_penalty = self.lambda_ready * readiness_loss

        # ── Halt penalty ───────────────────────────────────────────────
        # Penalise evidence emitted by active samples.
        # Mean over active → scale O(1), batch-size-independent.
        halt_penalty = self.lambda_halt * (evidence * active.float()).sum() / n_active

        # ── Tau loss (differentiable stopping time) ────────────────────
        # Only applied when lambda_tau > 0 and train_theta = True.
        # Gives theta a gradient path via dL/d(theta) = -lambda_tau / e_{t*}.
        # Masked to just-halted samples — tau is undefined for others.
        if self.lambda_tau > 0 and allow_threshold:
            n_halted    = just_halted.float().sum().clamp_min(1.0)
            tau_penalty = self.lambda_tau * (tau * just_halted.float()).sum() / n_halted
        else:
            tau_penalty = torch.tensor(0.0, device=lm_loss.device)

        total_loss = lm_loss + halt_penalty + readiness_penalty + tau_penalty

        # ── Metrics ────────────────────────────────────────────────────
        # No final-state metrics here (accuracy, steps, phi).
        # Those are read from carry after the episode loop in the training
        # script, exactly as ACT does — avoids all divide-by-zero drops.
        with torch.no_grad():
            metrics = {
                # Process metrics — accumulate over steps, divide by n_steps
                "lm_loss":      lm_loss.detach(),
                "halt_penalty": halt_penalty.detach(),
                "readiness":    readiness_penalty.detach(),
                "tau_penalty":  tau_penalty.detach(),

                # Evidence — sum and count kept separate for correct averaging
                "evidence_sum": (evidence.detach() * active.float()).sum(),
                "active_count": active.float().sum(),

                # Scalar — just take last value after episode loop
                "theta":        theta.detach(),
            }

        detached_outputs = (
            {k: outputs[k].detach() for k in return_keys if k in outputs}
            if return_keys else None
        )
        all_finish = new_carry.halted.all()

        return new_carry, total_loss, metrics, detached_outputs, all_finish