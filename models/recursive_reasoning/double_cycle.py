from contextlib import nullcontext
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

IGNORE_LABEL_ID = -100  


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
    H_cycles: int            # Number of recursion cycles (deep recursion steps - H cycles)
    L_cycles: int            # Number of reasoning cycles (latent recursion steps - L cycles)
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
    TRM High and low frequency recurrence:
    #TODO
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

        # State scales
        self.alpha_L = nn.Parameter(torch.full((), self.cfg.state_scale_init)) if self.cfg.scale_input_injection else 1
        self.alpha_H = nn.Parameter(torch.full((), self.cfg.state_scale_init)) if self.cfg.scale_input_injection else 1

        # State init
        self.ZL_init = nn.Parameter(torch.zeros(1, self.cfg.seq_len, d_model), requires_grad=False)
        self.ZH_init = nn.Parameter(torch.zeros(1, self.cfg.seq_len, d_model), requires_grad=False)


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

        param_device = next(self.parameters()).device
        if inputs.device != param_device:
            inputs = inputs.to(param_device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"[TRM] expected seq_len={self.cfg.seq_len}, got {L}"

        # Base embedding (tokens + positional)
        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)
        base_emb = self.token_emb(inputs) + self.pos_emb(pos)

        # Inital states
        Z_L = self.ZL_init.expand(B, L, -1).clone()
        Z_H = self.ZH_init.expand(B, L, -1).clone()

        logits_steps: List[torch.Tensor] = []
        H = max(1, self.cfg.H_cycles)
        L_inner = max(1, self.cfg.L_cycles)

        for h_step in range(H):
            # TBPTT
            ctx = torch.no_grad() if (self.cfg.detach_till_last and h_step < H - 1) else nullcontext()

            with ctx:
                # inner L cycles (latent recursion)
                for _ in range(L_inner):
                    Z_L = self.backbone(Z_L, Z_H + self.alpha_L * base_emb)

                # outer H update (deep recursion)
                Z_H = self.backbone(Z_H, self.alpha_H * Z_L)

                logits_t = self.head(Z_H)


            if return_all_logits:
                if h_step == H - 1:
                    logits_steps.append(logits_t)
                elif self.cfg.deep_supervision:
                    logits_steps.append(logits_t.detach())
        final_logits = logits_steps[-1] if logits_steps else logits_t
        return final_logits, (logits_steps if return_all_logits else None)

                    




