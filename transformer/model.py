# transformer/model.py
"""
Transformer — baseline reference implementation.

Paper: "Attention Is All You Need" (Vaswani et al., 2017)
https://arxiv.org/abs/1706.03762

Architecture
------------
Input tokens
     ↓
Token embedding + Positional encoding
     ↓
N × Encoder block:
   Multi-Head Self-Attention  (Q, K, V projections + scaled dot-product)
   Add & LayerNorm
   Feed-Forward Network       (d_model → d_ff → d_model, ReLU)
   Add & LayerNorm
     ↓
Linear projection → vocab logits

Key equations
-------------
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V          ← computed similarity
MultiHead(Q,K,V)   = Concat(head_1, …, head_h) · W_O
FFN(x)             = max(0, x W_1 + b_1) W_2 + b_2

Compare with Oscillator:
  dρ/dt = −i[H, ρ] + Σ_k D[c_k]ρ                       ← evolved dynamics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size:  int   = 32000
    max_seq_len: int   = 512
    d_model:     int   = 512      # embedding / hidden dimension
    n_heads:     int   = 8        # number of attention heads
    n_layers:    int   = 6        # number of encoder blocks
    d_ff:        int   = 2048     # feed-forward inner dimension
    dropout:     float = 0.1
    pad_id:      int   = 0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.d_k = self.d_model // self.n_heads   # per-head dimension


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: (B, h, T, d_k)
    mask:    (B, 1, 1, T)  or  (B, 1, T, T)  — True positions are masked out

    Returns
    -------
    out:    (B, h, T, d_k)
    attn:   (B, h, T, T)    attention weights (after softmax)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # (B,h,T,T)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    out = torch.matmul(attn, V)
    return out, attn


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.h    = cfg.n_heads
        self.d_k  = cfg.d_k
        self.d_model = cfg.d_model

        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (B, T, d_model)
        mask: optional padding or causal mask
        """
        B, T, _ = x.shape

        # Project and split into heads: (B, h, T, d_k)
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        out, attn = scaled_dot_product_attention(Q, K, V, mask, self.attn_dropout)

        # Concatenate heads: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), attn


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise FFN: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """One Transformer encoder layer: MHA → Add&Norm → FFN → Add&Norm."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn  = MultiHeadAttention(cfg)
        self.ff    = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Sub-layer 1: self-attention + residual
        attn_out, attn_w = self.attn(x, mask)
        x = self.norm1(x + self.drop(attn_out))

        # Sub-layer 2: FFN + residual
        x = self.norm2(x + self.drop(self.ff(x)))

        return x, attn_w


# ---------------------------------------------------------------------------
# Full Transformer (encoder-only, language model head)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Encoder-only Transformer with a language model head.

    Can be used for:
    - Masked language modeling (BERT-style)
    - Causal language modeling (with causal mask)
    - Sequence classification (pool the output)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.layers  = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie input/output embeddings (standard practice)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mask out padding tokens. Shape: (B, 1, 1, T)"""
        return (x == self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask. Shape: (1, 1, T, T)"""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        causal: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        tokens : (B, T)  integer token ids
        causal : bool    apply causal mask (for autoregressive generation)

        Returns
        -------
        dict with keys:
          logits      : (B, T, vocab_size)
          hidden      : (B, T, d_model)   final hidden states
          attn_weights: list of (B, h, T, T)  per-layer attention maps
        """
        B, T = tokens.shape

        # Build mask
        pad_mask = self.make_pad_mask(tokens)           # (B, 1, 1, T)
        if causal:
            causal_mask = self.make_causal_mask(T, tokens.device)   # (1,1,T,T)
            mask = pad_mask | causal_mask
        else:
            mask = pad_mask

        # Embed + positional encoding
        x = self.pos_enc(self.embed(tokens) * math.sqrt(self.cfg.d_model))

        # Encoder layers
        attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            attn_weights.append(attn_w)

        x = self.norm(x)
        logits = self.lm_head(x)

        return {
            "logits":       logits,
            "hidden":       x,
            "attn_weights": attn_weights,
        }

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        elif n >= 1_000_000:
            return f"{n/1e6:.1f}M"
        else:
            return f"{n/1e3:.1f}K"
