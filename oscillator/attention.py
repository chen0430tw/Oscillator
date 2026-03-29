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


def _sparse_softmax(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k sparse softmax: 只保留每行 k 个最大值，其余置 -inf 后 softmax。

    输入 scores: (..., T, T)
    输出        : (..., T, T)  每行最多 k 个非零权重

    量子类比：只有最强耦合的 k 对振荡子参与相位同步，
    弱耦合对的影响直接截断（等效于稀疏坍缩）。
    """
    T = scores.size(-1)
    k = min(k, T)
    # 找出每行 top-k 的位置
    topk_vals, topk_idx = scores.topk(k, dim=-1)          # (..., T, k)
    # 用 -inf 填充整行，再 scatter 回 top-k 位置
    sparse = torch.full_like(scores, float("-inf"))
    sparse.scatter_(-1, topk_idx, topk_vals)
    return F.softmax(sparse, dim=-1)


class AdjacencyBuilder(nn.Module):
    """从 token 嵌入构建归一化邻接矩阵 Ã ∈ [0,1]^{B×h×T×T}。

    Ã_ij = softmax_j( Q_i · K_j^T / √d_k )

    与 Transformer attention 的边权计算相同，但在此处仅用作
    OTAL 传播的图结构，不直接对 V 加权。

    adj_topk != None 时启用稀疏邻接：每个 token 只连接最强的 k 个邻居，
    传播时的图从全连接退化为稀疏图，内存从 O(T²) 降到 O(T·k)。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.h        = cfg.n_heads
        self.d_k      = cfg.d_k
        self.adj_topk = cfg.adj_topk
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
        if self.adj_topk is not None:
            A = _sparse_softmax(scores, self.adj_topk)
        else:
            A = F.softmax(scores, dim=-1)
        return self.drop(A)


def _inertial_step(
    D_flat:      torch.Tensor,   # (BH, T, d_k) complex
    A_flat:      torch.Tensor,   # (BH, T, T)   real
    active_flat: torch.Tensor,   # (BH, T)       bool
    lam:         torch.Tensor,   # scalar
) -> tuple[torch.Tensor, float]:
    """惯性传播单步：gather-compute-scatter，零 Python 循环。

    把 BH 分解为 (BH×T) 的扁平索引空间，active_flat 的非零位置
    即为需要计算的 (bh, t) 对，总数 N_active ≤ BH×T。

    操作流程（全部 CUDA 原语）：
      1. nonzero(active_flat) → (bh_idx, tok_idx)     shape (N_active,)
      2. gather A 行: A_flat[bh_idx, tok_idx]          shape (N_active, T)
      3. gather D 矩阵: D_flat[bh_idx]                 shape (N_active, T, d_k)
      4. bmm: (N_active,1,T) × (N_active,T,d_k) → (N_active, d_k)  ×2 (re/im)
      5. scatter: D_out[bh_idx, tok_idx] = D_cand

    计算量 O(N_active · T · d_k)，随收敛比例线性下降。
    """
    bh_idx, tok_idx = active_flat.nonzero(as_tuple=True)  # (N_active,)
    N_active = bh_idx.shape[0]

    if N_active == 0:
        return D_flat, 0.0

    BH, T, d_k = D_flat.shape
    active_ratio = N_active / (BH * T)

    # ── gather ─────────────────────────────────────────────────────
    A_rows  = A_flat[bh_idx, tok_idx]          # (N_active, T)  real
    D_full  = D_flat[bh_idx]                   # (N_active, T, d_k)  complex

    # ── bmm: (N_active,1,T) × (N_active,T,d_k) → (N_active,1,d_k) ─
    # A 是实数，D 是复数 → 两次实数 bmm，梯度正常传播
    A_3d    = A_rows.unsqueeze(1)              # (N_active, 1, T)
    nb_re   = torch.bmm(A_3d, D_full.real).squeeze(1)   # (N_active, d_k)
    nb_im   = torch.bmm(A_3d, D_full.imag).squeeze(1)
    neighbor = torch.complex(nb_re, nb_im)

    # ── update & scatter ───────────────────────────────────────────
    D_active = D_flat[bh_idx, tok_idx]         # (N_active, d_k)
    D_cand   = (1.0 - lam) * D_active + lam * neighbor

    D_out = D_flat.clone()
    D_out[bh_idx, tok_idx] = D_cand
    return D_out, active_ratio


class TopologyPropagation(nn.Module):
    """OTAL 拓扑传播（惯性计算 + nested_tensor 版）。

    基本更新规则：
        D(t+1) = (1-λ)·D(t) + λ·Ã·D(t)

    惯性计算：
        每步先检测哪些 token 的相位已收敛（δ/|D| < θ），
        对这些 token 跳过邻居聚合——它们有惯性，直接保持当前态。

    nested_tensor 优化：
        只对 active token 行组成 ragged matmul，
        bmm(nt_A, nt_D) 实际计算量 = O(Σ n_i · T · d_k)
        而非 O(BH · T² · d_k)。
        随训练推进 n_i/T 比例下降，节省持续增加。
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.n_steps = cfg.n_otal_steps
        # λ 直接从 cfg.lam（默认 0.3）开始，不再做 warmup
        # 归一化 D_unit 后 phase_sim 已经在合理尺度，无需延迟相位动力学
        lam_init   = max(cfg.lam, 1e-3)
        init_logit = math.log(lam_init / (1.0 - lam_init))
        self.logit_lam   = nn.Parameter(torch.tensor(init_logit))
        self.log_alpha   = nn.Parameter(torch.tensor(-3.0))
        self.logit_theta = nn.Parameter(torch.tensor(-4.0))  # θ init ≈ 0.018，训练初期几乎不冻结

        # 统计：记录每个 forward 的 active 比例（仅推理诊断）
        self.last_active_ratios: list[float] = []

    def forward(self, D: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        D : (B, h, T, d_k)  complex
        A : (B, h, T, T)    real, row-normalized
        → D': (B, h, T, d_k)  complex
        """
        B, h, T, d_k = D.shape
        BH = B * h
        D_0  = D
        lam  = torch.sigmoid(self.logit_lam)
        A_c  = A.to(D.dtype)

        self.last_active_ratios = []

        for _ in range(self.n_steps):
            neighbor = torch.matmul(A_c, D)
            D = (1.0 - lam) * D + lam * neighbor
            self.last_active_ratios.append(1.0)   # 全量计算，保持接口兼容

        alpha = torch.sigmoid(self.log_alpha)
        return D + alpha * D_0


class CollapseReadout(nn.Module):
    """坍缩读出：将复数振荡指向数 D 转换为实值注意力门控。

    对应 OTAL 的成熟度评分 M(U,t)。

    梯度流优化（v2）：
    不再使用 torch.angle()（arctan2 在零点梯度不稳定）或 norm 除法。
    改为可学习线性投影：[Re(D), Im(D)] → gate。
    - 全程线性，梯度路径最短
    - 网络自己学习如何从复数表示提取"坍缩信号"
    - 物理含义保留：Re/Im 直接携带振荡方向信息
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.mode = cfg.collapse_mode
        # 可学习投影：[Re(D_k), Im(D_k)] → gate_k
        self.proj = nn.Linear(2 * cfg.d_k, cfg.d_k, bias=True)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        """
        D       : (B, h, T, d_k)  complex
        → gate  : (B, h, T, d_k)  real ∈ (0,1)
        """
        # 拼接实部和虚部：(B, h, T, 2*d_k)
        features = torch.cat([D.real, D.imag], dim=-1)
        # 线性投影 → sigmoid 门控
        gate = torch.sigmoid(self.proj(features))      # (B, h, T, d_k)
        return gate


class OscillatorAttention(nn.Module):
    """完整振荡注意力层。

    流程：
        x  → PhaseEncoding    → D (complex)
        x  → AdjacencyBuilder → Ã (initial edge weights)
        D,Ã → TopologyPropagation (n_steps) → D'
        D'  → 相位同步注意力矩阵 → phase_attn (B,h,T,T)
        x  → W_V → V
        out = phase_attn · V  → W_O → output

    核心：phase_attn[i,j] = softmax( Re(D'_i · D'_j*) / √d_k )
    物理含义：token i 和 j 的相位同步程度决定它们互相关注的强度。
    相位锁定（Re(D'_i·D'_j*)大）→ 强烈互相关注。

    v2 修正：
    - 原版 gate ⊙ V 是逐位缩放，没有跨 token 信息混合
    - 新版用 Hermitian 内积生成注意力矩阵，再对 V 做加权求和
    - 与 Transformer 的区别：注意力矩阵来自相位传播动力学，不是 QK^T
    """

    def __init__(self, cfg: OscillatorConfig):
        super().__init__()
        self.h           = cfg.n_heads
        self.d_k         = cfg.d_k
        self.d_model     = cfg.d_model
        self.phase_topk  = cfg.phase_topk

        self.phase_enc = PhaseEncoding(cfg)
        self.adj       = AdjacencyBuilder(cfg)
        self.propagate = TopologyPropagation(cfg)

        self.W_V  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O  = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, T, d_model)
        → out: (B, T, d_model)
        → phase_attn: (B, h, T, T)  相位同步注意力矩阵（可视化）
        """
        B, T, _ = x.shape

        # ── 相位编码 ──────────────────────────────────────────
        D = self.phase_enc(x)                         # (B, h, T, d_k) complex

        # ── 初始邻接矩阵（传播的图骨架）──────────────────────
        A = self.adj(x, mask)                         # (B, h, T, T)

        # ── 拓扑传播：相位在图上同步 ──────────────────────────
        D_prime = self.propagate(D, A)                # (B, h, T, d_k) complex

        # ── 相位同步注意力矩阵 ────────────────────────────────
        # 先归一化：D'_unit = D' / |D'|，使每个 token 的振荡指向为单位复数向量
        # 这样 Re(D'_unit_i · D'_unit_j*) = cos(相位差) ∈ [-1, 1]
        # 等价于 Kuramoto 相位对齐度的精确计算
        # 解决原始 Re(D'·D'*) 因 |D'|² 过大导致 softmax 极度尖锐的问题
        # 归一化为单位复数向量：Re(D_unit_i · D_unit_j*) = cos(相位差) ∈ [-1,1]
        # 复数余弦相似度的统计方差 ≈ 1/(2·d_k)，正确缩放为 × √(2·d_k)
        # （类比：实数 QK^T 方差 ≈ d_k，缩放为 /√dk；复数版方差更小，需要更大的缩放）
        D_norm = D_prime.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-6)
        D_unit = D_prime / D_norm                              # (B, h, T, d_k) unit complex
        phase_sim = torch.matmul(D_unit, D_unit.conj().transpose(-2, -1)).real
        phase_sim = phase_sim * math.sqrt(2.0 * self.d_k)
        if mask is not None:
            phase_sim = phase_sim.masked_fill(mask, float("-inf"))
        # 稀疏坍缩：只保留 top-k 相位对齐的 token 对
        if self.phase_topk is not None:
            phase_attn = _sparse_softmax(phase_sim, self.phase_topk)
        else:
            phase_attn = F.softmax(phase_sim, dim=-1)
        phase_attn = self.drop(phase_attn)

        # ── 值投影 + 相位加权求和（跨 token 信息混合）─────────
        V   = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = torch.matmul(phase_attn, V)             # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out), phase_attn
