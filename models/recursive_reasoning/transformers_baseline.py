import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Tuple
from pydantic import BaseModel

IGNORE_LABEL_ID = -100  # if used by losses

class TransformerBaselineConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int  
    num_puzzle_identifiers: int = 1

    hidden_size: int
    num_heads: int
    expansion: float
    n_layers: int
    dropout: float
    pos_encodings: str = "learned"

    # ACT compatibility
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    no_ACT_continue: bool = True

    forward_dtype: str = "bfloat16"
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 0

class _EncoderOnly(nn.Module):
    def __init__(self, cfg: TransformerBaselineConfig):
        super().__init__()
        d_model = cfg.hidden_size
        d_ff = int(cfg.expansion * d_model)

        self.token_emb = nn.Embedding(cfg.vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(cfg.seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg.num_heads, dim_feedforward=d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.head    = nn.Linear(d_model, cfg.vocab_size)  # produce 11 logits (0 - <PAD>, 1 - blank, 2-10 - digits)

    def forward(self, inputs: torch.Tensor):
        B, L = inputs.shape
        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(inputs) + self.pos_emb(pos)
        h = self.encoder(h)
        logits = self.head(h)               # [B, 81, 11]
        return logits
    
    # --- wrapper to match TRM interface ---
class _DummyCarry:
    # simple placeholder to satisfy pretrain.py
    def __init__(self, batch_size, steps=0):
        self.steps  = torch.zeros(batch_size, dtype=torch.int32)
        self.halted = torch.ones(batch_size, dtype=torch.bool)
        self.inner_carry = None
        self.current_data = {}

class TransformerBaseline(nn.Module):
    """Baseline N-layer Transformer in the TRM/HRM wrapper interface."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TransformerBaselineConfig(**config_dict)
        self.net    = _EncoderOnly(self.config)

    @property
    def puzzle_emb(self):
        # not used; return None to keep compatibility
        return None

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        B = batch["inputs"].shape[0]
        c = _DummyCarry(B)
        # copy structure like TRM: keep a per-key tensor holder
        c.current_data = {k: torch.empty_like(v) for k,v in batch.items()}
        return c

    def forward(self, carry: _DummyCarry, batch: Dict[str, torch.Tensor]):
        # replace halted rows with fresh batch rows
        mask = carry.halted.view((-1,) + (1,) * (batch["inputs"].ndim - 1))
        current_data = {k: torch.where(mask, batch[k], carry.current_data.get(k, batch[k]))
                        for k in batch.keys()}
        # one pass (baseline does no ACT; halt after 1 step)
        logits = self.net(current_data["inputs"])

        outputs = {
            "logits": logits,  # [B, 81, 11]
            "q_halt_logits": torch.ones(logits.size(0), device=logits.device),
            "q_continue_logits": torch.zeros(logits.size(0), device=logits.device),
        }

        # advance steps; force halt (single-step baseline)
        steps  = carry.steps + 1
        halted = torch.ones_like(carry.halted)

        new_carry = _DummyCarry(logits.size(0))
        new_carry.steps  = steps
        new_carry.halted = halted
        new_carry.current_data = current_data
        return new_carry, outputs



