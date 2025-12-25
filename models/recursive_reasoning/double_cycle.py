from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

IGNORE_LABEL_ID = -100  

@dataclass
class TRMCarry:
    """Recurrent hidden state carried across reasoning episodes."""
    Z_S: torch.Tensor   # Solution latent (slow state)
    Z_R: torch.Tensor   # Reasoning latent (fast state)

class TRMConfig(BaseModel): 
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
    S_steps: int            # Number of solution recursion cycles (deep recursion steps - H cycles(legacy))
    R_steps: int            # Number of reasoning cycles (latent recursion steps - L cycles (legacy))
    detach_till_last: bool = True        # If True then we do no grad unlit H - 1 steps
    deep_supervision: bool = False       # If True, then we average CE loss across steps
    # residual_update: bool = True       # z_H <- z_H + ΔH   (otherwise z_H <- ΔH)
    scale_input_injection: bool = True   # If True, input embedding is scaled every L step
    state_scale_init: float = 0.1        # initial scale to add to input embed


class TRMBlock(nn.Module):
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        d_model = cfg.hidden_size
        d_ff = int(cfg.expansion * d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.num_heads,
            dim_feedforward=d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=cfg.n_layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        return self.encoder(hidden_states)

class TRM(nn.Module):
    """
    TRM model (no ACT here):
      - Two latents: reasoning state - Z_R (fast) and solution state Z_S (slow)
      - One shared tiny net (L_level) updates both states
      - Can be used in multi-episode training via TRMCarry
    """

    def __init__(self, cfg: Dict):
        super().__init__()

        # Config
        try:
            self.cfg = TRMConfig(**cfg)
        except ValidationError as e:
            raise ValueError(f"[TRM] bad config: {e}") from e

        # Embeddings
        d_model = self.cfg.hidden_size
        self.token_emb = nn.Embedding(self.cfg.vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(self.cfg.seq_len, d_model)

        # Reasoning net
        self.backbone = TRMBlock(self.cfg)

        # Classification head
        self.head = nn.Linear(d_model, self.cfg.vocab_size)

        # optional learnable gates for injection
        self.alpha_R = nn.Parameter(torch.full((), self.cfg.state_scale_init)) if self.cfg.scale_input_injection else 1
        self.alpha_S = nn.Parameter(torch.full((), self.cfg.state_scale_init)) if self.cfg.scale_input_injection else 1

        # state init buffers (non-trainable by default)
        self.ZR_init = nn.Parameter(torch.zeros(1, self.cfg.seq_len, d_model), requires_grad=False)
        self.ZS_init = nn.Parameter(torch.zeros(1, self.cfg.seq_len, d_model), requires_grad=False)

    def init_carry(self, batch_size: int, device: torch.device) -> TRMCarry:
        """Initialize Z_H and Z_L for a new puzzle episode."""
        Z_R = self.ZR_init.expand(batch_size, self.cfg.seq_len, -1).clone().to(device)
        Z_S = self.ZS_init.expand(batch_size, self.cfg.seq_len, -1).clone().to(device)
        return TRMCarry(Z_S=Z_S, Z_R=Z_R)

    def _base_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding."""
        B, L = inputs.shape
        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)
        return self.token_emb(inputs) + self.pos_emb(pos)

    def forward(
            self, 
            inputs: torch.Tensor, 
            carry: TRMCarry,
            *,
            return_all_logits: bool = True
            ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        One TRM reasoning episode:
          - runs S_steps outer updates and R_steps inner updates
          - updates carry (Z_S, Z_R)
          - returns final logits and optionally logits from intermediate solution steps

        Returns:
          new_carry, final_logits, logits_steps (optional)
        """

        param_device = next(self.parameters()).device
        if inputs.device != param_device:
            inputs = inputs.to(param_device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"[TRM] expected seq_len={self.cfg.seq_len}, got {L}"

        x_embed = self._base_embedding(inputs)
        Z_R = carry.Z_R
        Z_S = carry.Z_S
        logits_steps: List[torch.Tensor] = []
        solution_steps = max(1, self.cfg.S_steps)
        reasoning_steps = max(1, self.cfg.R_steps)

        for s_step in range(solution_steps):
            # TBPTT
            ctx = torch.no_grad() if (self.cfg.detach_till_last and s_step < solution_steps - 1) else nullcontext()

            with ctx:
                # inner reasoning cycles (latent recursion)
                for _ in range(reasoning_steps):
                    Z_R = self.backbone(Z_R, Z_S + self.alpha_R * x_embed)

                # outer solution update (deep recursion)
                Z_S = self.backbone(Z_S, self.alpha_S * Z_R)

                logits_t = self.head(Z_S)

            if return_all_logits:
                if s_step == solution_steps - 1:
                    logits_steps.append(logits_t)
                elif self.cfg.deep_supervision:
                    logits_steps.append(logits_t.detach())
        final_logits = logits_steps[-1] if logits_steps else logits_t
        new_carry = TRMCarry(Z_S=Z_S, Z_R=Z_R)
        return new_carry, final_logits, (logits_steps if return_all_logits else None)