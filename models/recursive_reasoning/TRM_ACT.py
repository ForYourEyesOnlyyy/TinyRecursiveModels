from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError
from models.recursive_reasoning.TRM_NoACT import TRMCarry, TRMConfig, TRM


from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)

IGNORE_LABEL_ID = -100  

@dataclass
class TRM_ACTCarry:
    """
    ACT wrapper carry (no streaming replacement):
      - inner_carry: Z_S/Z_R for each sample
      - steps: per-sample ACT step counter
      - halted: per-sample halted mask
      - last_logits: frozen logits for samples that have halted
    """
    inner_carry: "TRMCarry"
    steps: torch.Tensor         # [B] int32
    halted: torch.Tensor        # [B] bool
    last_logits: torch.Tensor   # [B, L, V] float32


class TRMACTConfig(TRMConfig):
    """
    TRMConfig + ACT fields (flat config).
    This lets you keep the same YAML keys and just add halt_* params.
    """
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True


class TRM_ACT(nn.Module):
    """
    TRM + ACT wrapper (no streaming replacement):
      - runs one TRM deep-reasoning step per call
      - updates per-sample halted mask using q_halt/q_continue (or q_halt>0)
      - freezes states/logits for halted samples
      - does NOT swap in new puzzles; caller loops until all halted and then loads next batch
    """

    def __init__(self, cfg_dict: Dict):
        super().__init__()
        self.config = TRMACTConfig(**cfg_dict)
        self.inner = TRM(cfg_dict) 

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRM_ACTCarry:
        """
        Initialize carry for a fresh batch. All samples start as active (halted=False).
        """
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device

        inner_carry = self.inner.init_carry(B, device)

        last_logits = torch.zeros(
            (B, self.config.seq_len, self.config.vocab_size),
            device=device,
            dtype=torch.float32,
        )

        return TRM_ACTCarry(
            inner_carry=inner_carry,
            steps=torch.zeros((B,), device=device, dtype=torch.int32),
            halted=torch.zeros((B,), device=device, dtype=torch.bool),
            last_logits=last_logits,
        )
    
    
    def forward(self, carry: TRM_ACTCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRM_ACTCarry, Dict[str, torch.Tensor]]:
        """
        One ACT step for the entire batch. No replacement occurs.

        Returns:
          new_carry, outputs dict with:
            - logits: [B,L,V] (frozen for halted samples)
            - q_halt_logits: [B]
            - q_continue_logits: [B]
            - halted: [B] bool
            - steps: [B] int
        """
        inputs = batch["inputs"]

        new_inner_carry, logits = self.inner(inputs, carry.inner_carry)  # logits [B,L,V]
        logits = logits.to(torch.float32)

        q_logits = self.q_head(new_inner_carry.Z_S[:, 0]).to(torch.float32)  # [B,2]
        q_halt_logits = q_logits[:, 0]
        q_continue_logits = q_logits[:, 1]

        with torch.no_grad():
            prev_halted = carry.halted
            steps = torch.where(prev_halted, carry.steps, carry.steps + 1)

            halted = steps >= self.config.halt_max_steps

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                if self.config.halt_exploration_prob > 0:
                    rand = torch.rand_like(q_halt_logits)
                    min_halt_steps = (rand < self.config.halt_exploration_prob) * torch.randint_like(
                        steps, low=2, high=self.config.halt_max_steps + 1
                    )
                    halted = halted & (steps >= min_halt_steps)

            halted = prev_halted | halted
            just_halted = (~prev_halted) & halted

        freeze_mask = prev_halted.view(-1, 1, 1)  # [B,1,1]

        frozen_logits = torch.where(freeze_mask, carry.last_logits, logits)
        ZS = torch.where(freeze_mask, carry.inner_carry.Z_S, new_inner_carry.Z_S)
        ZR = torch.where(freeze_mask, carry.inner_carry.Z_R, new_inner_carry.Z_R)

        new_carry = TRM_ACTCarry(
            inner_carry=TRMCarry(Z_S=ZS, Z_R=ZR),
            steps=steps,
            halted=halted,
            last_logits=frozen_logits,
        )

        outputs = {
            "logits": frozen_logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "just_halted": just_halted,
        }
        return new_carry, outputs