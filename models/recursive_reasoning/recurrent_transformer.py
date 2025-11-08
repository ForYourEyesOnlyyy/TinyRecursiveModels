from contextlib import nullcontext
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

IGNORE_LABEL_ID = -100  # if used by losmodels/recursive_reasoning/transformers_baseline.pyses


class HRecTransformerConfig(BaseModel): 
    # data / vocab
    seq_len: int                    # 81 for Sudoku
    vocab_size: int                 # 11: {0=PAD, 1=blank, 2..10=digits 1..9}

    # backbone
    hidden_size: int                # d_model (512)
    num_heads: int                  # 8
    expansion: float                # FFN scale (4.0 -> 2048)
    n_layers: int                   # number of encoder layers (6)
    dropout: float = 0.1

    # recursion
    recursion_steps: int            # Number of H cycles
    detach_till_last: bool = True   # If True then we do no grad unlit H - 1 steps
    deep_supervision: bool = False  # If True, then we average CE loss across steps
    residual_update: bool = True    # z_H <- z_H + ΔH   (otherwise z_H <- ΔH)
    state_scale_init: float = 0.1   # initial scale to add to input embed


class HRecTransformer(nn.Module):
    """
    H-only recurrence:
      - tokens are fixed
      - slow solution state z_H (B, L, D) is updated across steps
      - embeddings_t = token_emb(x) + pos_emb + alpha * z_H_t
      - encoder(shared) -> h_t; z_H_{t+1} = z_H_t (+) h_t
      - logits are decoded from z_H (shared head for all steps)

    Weights are shared across steps and z_H provides feedback, 
    enabling iterative refinement with TBPTT-style training.
    """

    def __init__(self, cfg: dict):
        try:
            self.cfg = HRecTransformerConfig(**cfg)
        except ValidationError as e:
            raise ValueError(f"[HRecTransformer] bad config: {e}") from e

        # copy/paste from baseline
        d_model = self.cfg.hidden_size
        d_ff = int(self.cfg.expansion * d_model)
        self.token_emb = nn.Embedding(self.cfg.vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(self.cfg.seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.cfg.num_heads,
            dim_feedforward=d_ff,
            dropout=self.cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=self.cfg.n_layers)
        self.head = nn.Linear(d_model, self.cfg.hidden_size)

        # recursion
        self.alpha = nn.Parameter(torch.full((), self.cfg.state_scale_init))

        # optional
        self.Z_init = nn.Parameter(torch.zeros(1, self.cfg.seq_len, d_model), requires_grad=False)

    def forward(
            self, 
            inputs: torch.Tensor, 
            *, 
            return_all_logits: bool = True
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        inputs: [B, L] long token ids
        return:
          - final_logits: [B, L, V]
          - logits_steps: Optional[List[[B, L, V]]] if return_all_logits=True
        """

        # copy/paste from baseline
        param_device = next(self.parameters()).device
        if inputs.device != param_device:
            inputs = inputs.to(param_device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"expected seq_len={self.cfg.seq_len}, got {L}"
        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)

        # new stuff
        Z = self.Z_init.expand(B, L, -1).clone()
        logits_steps: List[torch.Tensor] = []
        T = max(1, self.cfg.recursion_steps)
        base_emb = self.token_emb(inputs) + self.pos_emb(pos)

        # now do recursion
        if self.cfg.detach_till_last and T > 1:
            ctx = torch.no_grad()
        else:
            ctx = nullcontext()
        
        with ctx:
            for step in range(T - 1):
                emb = base_emb + self.alpha * Z
                h = self.encoder(emb)

                if self.cfg.residual_update:
                    Z = Z + h
                else:
                    Z = h

                logits_t = self.head(Z)

                if return_all_logits:
                    logits_steps.append(logits_t.detach() if self.cfg.detach_till_last else logits_t)
        
        # always with grad the last step
        emb = self.token_emb(inputs) + self.pos_emb(pos) + self.alpha * Z
        h = self.encoder(emb)
        if self.cfg.residual_update:
            Z = Z + h
        else:
            Z = h
        final_logits = self.head(Z)

        if return_all_logits:
            logits_steps.append(final_logits)
        
        return final_logits, (logits_steps if return_all_logits else None)

                    




