from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

from models.common import trunc_normal_init_
from models.layers import CastedEmbedding, CastedLinear

IGNORE_LABEL_ID = -100  

@dataclass
class TRMCarry:
    """Recurrent hidden state carried across reasoning episodes."""
    Z_S: torch.Tensor   # Solution latent (slow state)
    Z_R: torch.Tensor   # Reasoning latent (fast state)

class TRMConfig(BaseModel): 
    # data / vocab
    forward_dtype: str = "bfloat16"
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
    scale_input_injection: bool = True   # If True, input embedding is scaled every L step
    state_scale_init: float = 0.1        # initial scale to add to input embed


class TRMBlock(nn.Module):
    """
    Shared tiny network used for both Z_R and Z_S updates:
        hidden_states <- hidden_states + input_injection
        hidden_states <- TransformerEncoder(hidden_states)
    """
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

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
            
        self.forward_dtype = getattr(torch, self.cfg.forward_dtype)

        # Embeddings
        d_model = self.cfg.hidden_size
        self.embed_scale = math.sqrt(d_model)

        embed_init_std = 1.0 / self.embed_scale

        # Casted embeddings + head 
        self.token_emb = CastedEmbedding(
            self.cfg.vocab_size,
            d_model,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.pos_emb = CastedEmbedding(
            self.cfg.seq_len,
            d_model,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.head = CastedLinear(d_model, self.cfg.vocab_size, bias=True)

        # Shared tiny net
        self.backbone = TRMBlock(self.cfg)

        # optional learnable gates for injection
        if self.cfg.scale_input_injection:
            self.alpha_R = nn.Parameter(torch.full((), float(self.cfg.state_scale_init)))
            self.alpha_S = nn.Parameter(torch.full((), float(self.cfg.state_scale_init)))
        else:
            self.register_buffer("alpha_R", torch.tensor(1.0), persistent=True)
            self.register_buffer("alpha_S", torch.tensor(1.0), persistent=True)

        # state init buffers (non-trainable by default)
        self.register_buffer(
            "ZS_init_vec",
            trunc_normal_init_(torch.empty(d_model), std=1.),
            persistent=True,
        )
        self.register_buffer(
            "ZR_init_vec",
            trunc_normal_init_(torch.empty(d_model), std=1.),
            persistent=True,
        )

    def _base_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding, scaled."""
        B, L = inputs.shape
        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)
        return self.embed_scale * (self.token_emb(inputs) + self.pos_emb(pos))

    def init_carry(self, batch_size: int, device: torch.device) -> TRMCarry:
        """
        fixed random init:
        broadcast [D] init vectors to [B, L, D].
        """
        Z_S = self.ZS_init_vec.to(device).view(1, 1, -1).expand(batch_size, self.cfg.seq_len, -1).clone()
        Z_R = self.ZR_init_vec.to(device).view(1, 1, -1).expand(batch_size, self.cfg.seq_len, -1).clone()
        return TRMCarry(Z_S=Z_S, Z_R=Z_R)

    def forward(
        self,
        inputs: torch.Tensor,
        carry: TRMCarry,
    ) -> Tuple[TRMCarry, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        One TRM reasoning episode:
          - inner: R_steps updates of Z_R using (Z_S + x_embed)
          - outer: 1 update of Z_S using Z_R
        Deep recursion (outer loop) uses repo-style mandatory BPTT:
          - S_steps-1 outer iterations under no_grad
          - final outer iteration with gradients
        Returns:
          new_carry (detached), final_logits
        """
        device = next(self.parameters()).device
        if inputs.device != device:
            inputs = inputs.to(device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"[TRM] expected seq_len={self.cfg.seq_len}, got {L}"

        x_embed = self._base_embedding(inputs)

        Z_S = carry.Z_S
        Z_R = carry.Z_R

        S = max(1, int(self.cfg.S_steps))
        R = max(1, int(self.cfg.R_steps))

        # S-1 steps with no grad, last step with grad.
        with torch.no_grad():
            for s_step in range(S - 1):
                for _ in range(R):
                    Z_R = self.backbone(Z_R, Z_S + self.alpha_R * x_embed)
                Z_S = self.backbone(Z_S, self.alpha_S * Z_R)

        # Final outer step WITH gradients
        for _ in range(R):
            Z_R = self.backbone(Z_R, Z_S + self.alpha_R * x_embed)
        Z_S = self.backbone(Z_S, self.alpha_S * Z_R)

        final_logits = self.head(Z_S)

        # detach carry before returning
        new_carry = TRMCarry(Z_S=Z_S.detach(), Z_R=Z_R.detach())
        return new_carry, final_logits