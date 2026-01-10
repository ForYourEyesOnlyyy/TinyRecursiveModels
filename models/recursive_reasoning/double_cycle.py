from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ValidationError

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

    pos_encodings: str = "rope"      # only "rope" is supported here
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5


class TRMBlock(nn.Module):
    """
    Tiny block:
      - Self-attention with RoPE
      - SwiGLU FFN
      - RMSNorm post-norm residuals
    """
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0, "hidden_size must be divisible by num_heads"
        head_dim = cfg.hidden_size // cfg.num_heads

        self.attn = Attention(
            hidden_size=cfg.hidden_size,
            head_dim=head_dim,
            num_heads=cfg.num_heads,
            num_key_value_heads=cfg.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(hidden_size=cfg.hidden_size, expansion=cfg.expansion)
        self.eps = cfg.rms_norm_eps

    def forward(self, hidden_states: torch.Tensor, *, cos_sin: CosSin) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.attn(hidden_states=hidden_states, cos_sin=cos_sin),
            variance_epsilon=self.eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.eps,
        )
        return hidden_states


class TRMReasoningModule(nn.Module):
    """
    Shared tiny net used for both Z_R and Z_S updates:
      hidden <- hidden + injection
      hidden <- [TRMTinyBlock] x n_layers
    """
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.layers = nn.ModuleList([TRMBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, *, cos_sin: CosSin) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states

class TRM(nn.Module):
    """
    TRM without ACT:
      - Two latents: Z_R (fast reasoning) and Z_S (slow solution)
      - Shared tiny net with RoPE attention
      - Mandatory fixed random init buffers
      - Mandatory repo-style BPTT:
          S_steps-1 outer steps under no_grad
          last outer step with gradients
      - Optional gating
    """
    def __init__(self, cfg_dict: Dict):
        super().__init__()
        try:
            self.cfg = TRMConfig(**cfg_dict)
        except ValidationError as e:
            raise ValueError(f"[TRM] invalid config: {e}") from e

        assert self.cfg.pos_encodings == "rope", "This TRM version supports only pos_encodings='rope'"

        self.forward_dtype = getattr(torch, self.cfg.forward_dtype)
        D = self.cfg.hidden_size

        # embedding scaling (repo-style)
        self.embed_scale = math.sqrt(D)
        embed_init_std = 1.0 / self.embed_scale

        # Casted embeddings + head (repo-style)
        self.token_emb = CastedEmbedding(
            self.cfg.vocab_size,
            D,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )

        self.head = CastedLinear(D, self.cfg.vocab_size, bias=True)

        # RoPE generator (repo-style)
        self.rotary_emb = RotaryEmbedding(
            dim=D // self.cfg.num_heads,
            max_position_embeddings=self.cfg.seq_len,
            base=self.cfg.rope_theta,
        )

        # Shared tiny net
        self.reasoning = TRMReasoningModule(self.cfg)


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
            trunc_normal_init_(torch.empty(D), std=1.),
            persistent=True,
        )
        self.register_buffer(
            "ZR_init_vec",
            trunc_normal_init_(torch.empty(D), std=1.),
            persistent=True,
        )

    def _token_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        # CastedEmbedding expects int32 ids
        tok = self.token_emb(inputs.to(torch.int32))
        return self.embed_scale * tok

    def init_carry(self, batch_size: int, device: torch.device) -> TRMCarry:
        Z_S = self.ZS_init_vec.to(device).view(1, 1, -1).expand(batch_size, self.cfg.seq_len, -1).clone()
        Z_R = self.ZR_init_vec.to(device).view(1, 1, -1).expand(batch_size, self.cfg.seq_len, -1).clone()
        return TRMCarry(Z_S=Z_S, Z_R=Z_R)

    def forward(self, inputs: torch.Tensor, carry: TRMCarry) -> Tuple[TRMCarry, torch.Tensor]:
        device = next(self.parameters()).device
        if inputs.device != device:
            inputs = inputs.to(device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"[TRM] expected seq_len={self.cfg.seq_len}, got {L}"

        cos_sin = self.rotary_emb()  # RoPE cos/sin for attention
        x_embed = self._token_embedding(inputs)  # [B, L, D]

        Z_S = carry.Z_S
        Z_R = carry.Z_R

        S = max(1, int(self.cfg.S_steps))
        R = max(1, int(self.cfg.R_steps))

        # BPTT: S-1 outer steps without gradients
        with torch.no_grad():
            for _ in range(S - 1):
                for _ in range(R):
                    Z_R = self.reasoning(Z_R, Z_S + self.alpha_R * x_embed, cos_sin=cos_sin)
                Z_S = self.reasoning(Z_S, self.alpha_S * Z_R, cos_sin=cos_sin)

        # Final outer step WITH gradients
        for _ in range(R):
            Z_R = self.reasoning(Z_R, Z_S + self.alpha_R * x_embed, cos_sin=cos_sin)
        Z_S = self.reasoning(Z_S, self.alpha_S * Z_R, cos_sin=cos_sin)

        logits = self.head(Z_S)

        new_carry = TRMCarry(Z_S=Z_S.detach(), Z_R=Z_R.detach())
        return new_carry, logits