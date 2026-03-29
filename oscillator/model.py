# oscillator/model.py
"""
Oscillator — 完整模型

架构：
    Input tokens
         ↓
    Token embedding + Positional encoding
         ↓
    N × OscillatorBlock:
       OscillatorAttention   ← 相位传播 + 坍缩读出（取代 MultiHeadAttention）
       Add & LayerNorm
       FeedForward            ← 与 Transformer 相同
       Add & LayerNorm
         ↓
    LayerNorm
         ↓
    LM Head (tied embeddings)

核心差异：
    Transformer  attention = softmax(QKᵀ/√d) · V        (显式计算)
    Oscillator   attention = collapse(propagate(D, Ã))    (动力学涌现)

    D_i(t+1) = (1-λ)·D_i(t) + λ·Σ_j Ã_ij·D_j(t)
    ← 这就是 OTAL 的振荡指向数传播，在此作为注意力层的核心运算
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OscillatorConfig
from .attention import OscillatorAttention


# ── Positional Encoding（与 Transformer 相同）──────────────────────────

class PositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── FeedForward（与 Transformer 相同）──────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── OscillatorBlock ────────────────────────────────────────────────────

class OscillatorBlock(nn.Module):
    """单个 Oscillator 层。

    OscillatorAttention → Add & LayerNorm → FeedForward → Add & LayerNorm

    与 Transformer EncoderBlock 的唯一区别：
        MultiHeadAttention → OscillatorAttention
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.attn  = OscillatorAttention(cfg)
        self.ff    = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, gate = self.attn(x, mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, gate


# ── Oscillator（完整模型）─────────────────────────────────────────────

class Oscillator(nn.Module):
    """振荡器语言模型。

    基于相位动力学的 Transformer 替代架构：
    注意力权重不由 Q·K^T 算出，而是由振荡指向数的传播动力学涌现。

    Parameters
    ----------
    cfg : OscillatorConfig

    Notes
    -----
    与 Transformer 的参数量对比（相同 d_model/n_heads）：
    - 多出：W_re, W_im（相位编码）= 2 × d_model²
    - 少出：无。W_Q/W_K/W_V/W_O 都保留（W_Q,W_K 用于邻接，W_V,W_O 用于输出）
    - 多出：logit_lam（每层 1 个可学习标量）
    - 净增：每层多 2 × d_model² 参数（相位编码层）
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.cfg = cfg

        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.layers  = nn.ModuleList([OscillatorBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight   # tie embeddings
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.is_floating_point() and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()\
                     .unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        causal: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        tokens : (B, T)
        causal : bool

        Returns
        -------
        dict:
            logits       : (B, T, vocab_size)
            hidden       : (B, T, d_model)
            gates        : list of (B, h, T, d_k)  — 每层坍缩门控（可视化用）
            lam_values   : list of float — 每层当前耦合系数 λ
        """
        B, T = tokens.shape

        pad_mask = self.make_pad_mask(tokens)
        if causal:
            mask = pad_mask | self.make_causal_mask(T, tokens.device)
        else:
            mask = pad_mask

        x = self.pos_enc(self.embed(tokens) * math.sqrt(self.cfg.d_model))

        gates      = []
        lam_values = []
        for layer in self.layers:
            x, gate = layer(x, mask)
            gates.append(gate)
            lam_values.append(
                torch.sigmoid(layer.attn.propagate.logit_lam).item()
            )

        x      = self.norm(x)
        logits = self.lm_head(x)

        return {
            "logits":     logits,
            "hidden":     x,
            "gates":      gates,
            "lam_values": lam_values,
        }

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        elif n >= 1_000_000:
            return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"
