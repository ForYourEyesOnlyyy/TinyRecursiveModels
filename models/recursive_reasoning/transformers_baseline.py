import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Tuple
from pydantic import BaseModel

IGNORE_LABEL_ID = -100  # if used by losmodels/recursive_reasoning/transformers_baseline.pyses

class SudokuTransformerConfig(BaseModel):
    # data / vocab
    seq_len: int                    # 81 for Sudoku
    vocab_size: int                 # 11: {0=PAD, 1=blank, 2..10=digits 1..9}

    # backbone
    hidden_size: int                # d_model (512)
    num_heads: int                  # 8
    expansion: float                # FFN scale (4.0 -> 2048)
    n_layers: int                   # number of encoder layers (6)
    dropout: float = 0.1


class SudokuTransformer(nn.Module):
    """
    Plain encoder-only Transformer for Sudoku (or any token sequence).
    Forward signature: logits = model(inputs)
    - inputs: LongTensor [B, L] with tokens in [0..vocab_size-1]
              (your convention: 1=blank, 2..10=digits)
    - logits: FloatTensor [B, L, vocab_size]
    """

    def __init__(self, cfg: SudokuTransformerConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_size
        d_ff = int(cfg.expansion * d_model)
        self.token_emb = nn.Embedding(cfg.vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(cfg.seq_len, d_model)

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
        self.head = nn.Linear(d_model, cfg.hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, L] long; logits: [B, L, V]
        Ensures inputs are on the same device as the module parameters (MPS-safe).
        """
        param_device = next(self.parameters()).device
        if inputs.device != param_device:
            inputs = inputs.to(param_device)

        B, L = inputs.shape
        assert L == self.cfg.seq_len, f"expected seq_len={self.cfg.seq_len}, got {L}"

        pos = torch.arange(L, device=inputs.device).unsqueeze(0).expand(B, L)
        h = self.token_emb(inputs) + self.pos_emb(pos)
        h = self.encoder(h)
        logits = self.head(h)

        return logits


