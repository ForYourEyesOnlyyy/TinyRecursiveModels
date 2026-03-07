from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    """
    Fixed-batch ACT loss head.
    """

    def __init__(self, model: nn.Module, loss_type: str = "stablemax_cross_entropy"):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]  # stablemax_cross_entropy / softmax_cross_entropy

    def forward(
        self,
        *,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        One ACT step:
          - calls model(carry,batch) once
          - computes LM loss + Q losses for this step
          - returns:
              new_carry,
              loss (scalar),
              metrics (scalars, reduced-ready),
              detached_outputs (optional),
              all_finish (bool tensor: new_carry.halted.all())
        """
        new_carry, outputs = self.model(carry=carry, batch=batch)  # type: ignore
        labels = batch["labels"]


        with torch.no_grad():
            preds = torch.argmax(outputs["logits"], dim=-1)

            mask = (labels != IGNORE_LABEL_ID)                 # [B,L]
            loss_counts = mask.sum(-1)                         # [B]
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # [B,1]

            is_correct = mask & (preds == labels)              # [B,L]
            seq_is_correct = (is_correct.sum(-1) == loss_counts)  # [B] bool
            just_halted = outputs.get("just_halted", None)
            if just_halted is None:
                just_halted = new_carry.halted

            valid_metrics = just_halted & (loss_counts > 0)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0.0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
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
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "q_continue_loss": q_continue_loss.detach(),
        })

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs} if return_keys else None
        all_finish = new_carry.halted.all()
        return new_carry, total_loss, metrics, detached_outputs, all_finish
    

class RTifyLossHead(nn.Module):

    def __init__(self, model, loss_type="stablemax_cross_entropy"):
        super().__init__()

        self.model = model
        self.loss_fn = globals()[loss_type]

        self.lambda_halt = model.config.lambda_halt


    def forward(
        self,
        *,
        carry,
        batch,
        return_keys=(),
    ):

        new_carry, outputs = self.model(carry=carry, batch=batch)

        labels = batch["labels"]
        logits = outputs["logits"]
        evidence = outputs["evidence"]
        phi = outputs["phi"]
        theta = outputs["theta"]

        mask = (labels != IGNORE_LABEL_ID)

        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        lm_loss = (
            self.loss_fn(
                logits,
                labels,
                ignore_index=IGNORE_LABEL_ID,
                valid_mask=mask,
            ) / loss_divisor
        ).sum()

        halt_penalty = self.lambda_halt * evidence.sum()
        total_loss = lm_loss + halt_penalty

        with torch.no_grad():

            preds = logits.argmax(-1)

            is_correct = mask & (preds == labels)

            seq_is_correct = (
                is_correct.sum(-1) == loss_counts
            )

            just_halted = outputs["just_halted"]

            valid_metrics = just_halted & (loss_counts > 0)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.float() / loss_divisor).sum(-1),
                    0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "lm_loss": lm_loss.detach(),
                "halt_penalty": halt_penalty.detach(),
                "steps_sum": torch.where(
                    valid_metrics,
                    new_carry.steps.float(),
                    0
                ).sum(),

                "phi_sum": torch.where(
                    valid_metrics,
                    phi,
                    0
                ).sum(),

                "theta": theta.detach(),
            }

        detached_outputs = {
            k: outputs[k].detach()
            for k in return_keys
            if k in outputs
        }

        all_finish = new_carry.halted.all()

        return new_carry, total_loss, metrics, detached_outputs, all_finish