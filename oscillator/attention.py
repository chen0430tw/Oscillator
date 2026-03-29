# oscillator/attention.py
"""
OscillatorAttention — 振荡注意力机制

Transformer 的注意力是手工计算的：
    score(Q,K) = softmax(QKᵀ / √d_k) · V             ← 一次性计算

Oscillator 的注意力是从动力学中涌现的：
    D_i(t+Δt) = (1-λ)·D_i(t) + λ·Σ_j Ã_ij·D_j(t)   ← 迭代传播
    gate_i     = alignment(D_i, mean(D))               ← 坍缩读出

这里的 D_i 就是 OTAL 的振荡指向数（complex）。
边权矩阵 Ã 就是 OTAL 的动态边权（由 token 内积学习）。
坍缩读出就是 OTAL 的成熟度评分（Kuramoto 相位对齐度）。

OTAL 是 OscillatorAttention 在图上的原型实现，
OscillatorAttention 是 OTAL 的可微分矩阵形式。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OscillatorConfig


class PhaseEncoding(nn.Module):
    """将实值 token 嵌入投影为复数振荡指向数 D ∈ ℂ^{B×h×T×d_k}。

    D = W_re·x + i·W_im·x

    这是相位编码步骤：把 token 的语义信息编码进相位空间。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.h   = cfg.n_heads
        self.d_k = cfg.d_k
        # 实部和虚部分别投影（不共享权重）
        self.W_re = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_im = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        → D: (B, h, T, d_k)  complex64/128
        """
        B, T, _ = x.shape
        re = self.W_re(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        im = self.W_im(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        return torch.complex(re, im)


class AdjacencyBuilder(nn.Module):
    """从 token 嵌入构建归一化邻接矩阵 Ã ∈ [0,1]^{B×h×T×T}。

    Ã_ij = softmax_j( Q_i · K_j^T / √d_k )

    与 Transformer attention 的边权计算相同，但在此处仅用作
    OTAL 传播的图结构，不直接对 V 加权。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.h   = cfg.n_heads
        self.d_k = cfg.d_k
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (B, 1, 1, T) or (B, 1, T, T)
        → Ã : (B, h, T, T)
        """
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        A = F.softmax(scores, dim=-1)
        return self.drop(A)


class TopologyPropagation(nn.Module):
    """OTAL 拓扑传播（可微分矩阵形式）。

    n_steps 步更新规则：
        D(t+1) = (1-λ)·D(t) + λ·Ã·D(t)
               = ((1-λ)·I + λ·Ã) · D(t)

    这是 OTAL topology_update.py 的矩阵形式，全程可微分。
    λ 是可学习参数（通过 sigmoid 约束在 (0,1)）。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.n_steps = cfg.n_otal_steps
        # 可学习耦合系数（初始化为 cfg.lam 对应的 logit）
        import math as _math
        init_logit = _math.log(cfg.lam / (1.0 - cfg.lam))
        self.logit_lam = nn.Parameter(torch.tensor(init_logit))

    def forward(self, D: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        D : (B, h, T, d_k)  complex
        A : (B, h, T, T)    real, row-normalized
        → D': (B, h, T, d_k)  complex
        """
        lam = torch.sigmoid(self.logit_lam)
        A_c = A.to(D.dtype)   # cast adjacency to complex for matmul
        for _ in range(self.n_steps):
            # 邻域加权和：Ã · D
            neighbor = torch.matmul(A_c, D)        # (B, h, T, d_k)
            D = (1.0 - lam) * D + lam * neighbor
        return D


class CollapseReadout(nn.Module):
    """坍缩读出：将复数振荡指向数 D 转换为实值注意力门控。

    对应 OTAL 的成熟度评分 M(U,t)。

    phase_align 模式（Kuramoto 相位对齐）：
        mean_D  = mean(D, dim=T)              # 平均振荡方向（均场）
        gate_i  = Re(D_i · conj(mean_D)) / (|D_i|·|mean_D| + ε)
                  ∈ [-1, 1]，1 表示与均场完全对齐
        weights = softmax(gate, dim=T)        # 归一化为注意力权重

    amplitude 模式（振幅坍缩）：
        weights = softmax(|D|², dim=T)

    注意：这里不是"算出"注意力——注意力权重是 n_steps 迭代后
    自然涌现出的相位分布的读出，等同于量子态测量。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.mode = cfg.collapse_mode

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        """
        D       : (B, h, T, d_k)  complex
        → gate  : (B, h, T, d_k)  real, softmax-normalized over T
        """
        if self.mode == "phase_align":
            # 均场方向
            mean_D = D.mean(dim=-2, keepdim=True)              # (B, h, 1, d_k)
            # 相位对齐度：Re(D · mean_D*) / (|D|·|mean_D| + ε)
            dot    = (D * mean_D.conj()).real                   # (B, h, T, d_k)
            norm   = D.abs() * mean_D.abs() + 1e-8
            align  = dot / norm                                 # ∈ [-1, 1]
            # softmax over token dim → 注意力权重
            gate   = F.softmax(align, dim=-2)                  # (B, h, T, d_k)
        else:  # amplitude
            gate   = F.softmax(D.abs() ** 2, dim=-2)

        return gate


class OscillatorAttention(nn.Module):
    """完整振荡注意力层。

    流程：
        x  → PhaseEncoding  → D (complex)
        x  → AdjacencyBuilder → Ã (real, row-norm)
        D,Ã → TopologyPropagation (n_steps) → D'
        D'  → CollapseReadout → gate (real)
        x  → W_V → V
        out = gate ⊙ V  → W_O → output

    对比 Transformer：
        MultiHeadAttention: out = softmax(QKᵀ/√d) · V    (一步计算)
        OscillatorAttention: out = collapse(propagate(D,Ã)) ⊙ V  (动力学涌现)
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.h       = cfg.n_heads
        self.d_k     = cfg.d_k
        self.d_model = cfg.d_model

        self.phase_enc = PhaseEncoding(cfg)
        self.adj       = AdjacencyBuilder(cfg)
        self.propagate = TopologyPropagation(cfg)
        self.readout   = CollapseReadout(cfg)

        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, T, d_model)
        → out: (B, T, d_model)
        → gate: (B, h, T, d_k)  可视化用
        """
        B, T, _ = x.shape

        # ── 相位编码 ─────────────────────────────────────────
        D = self.phase_enc(x)                        # (B, h, T, d_k) complex

        # ── 邻接矩阵（OTAL 边权）─────────────────────────────
        A = self.adj(x, mask)                        # (B, h, T, T)

        # ── 拓扑传播（OTAL 核心）─────────────────────────────
        D_prime = self.propagate(D, A)               # (B, h, T, d_k) complex

        # ── 坍缩读出（相位对齐 → 注意力门控）────────────────
        gate = self.readout(D_prime)                 # (B, h, T, d_k) real

        # ── 值投影 + 门控输出 ─────────────────────────────────
        V   = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = self.drop(gate) * V                    # 相位门控（不是加权求和）
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), gate
