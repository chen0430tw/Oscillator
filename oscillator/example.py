# oscillator/example.py
"""
Oscillator vs Transformer 对比验证。

用法：
    py -3.13 oscillator/example.py
"""

from __future__ import annotations

import sys, math
sys.path.insert(0, ".")

import torch
from oscillator.model import Oscillator
from oscillator.config import OscillatorConfig
from transformer.model import Transformer, TransformerConfig


def main():
    # ── 相同规模配置 ──────────────────────────────────────────
    common = dict(
        vocab_size=1000, max_seq_len=64,
        d_model=128, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.0,
    )

    t_cfg   = TransformerConfig(**common)
    osc_cfg = OscillatorConfig(**common, n_otal_steps=3, lam=0.3)

    transformer = Transformer(t_cfg)
    oscillator  = Oscillator(osc_cfg)

    print("=" * 60)
    print(f"Transformer  params: {transformer.param_count()}")
    print(f"Oscillator   params: {oscillator.param_count()}")
    print("  (Oscillator 多出 W_re, W_im 相位编码层)")
    print("=" * 60)

    # ── 前向传播 ──────────────────────────────────────────────
    B, T = 2, 16
    tokens = torch.randint(1, 1000, (B, T))
    tokens[0, -3:] = 0   # padding

    t_out   = transformer(tokens, causal=False)
    osc_out = oscillator(tokens, causal=False)

    print(f"\nTransformer forward pass:")
    print(f"  logits : {t_out['logits'].shape}")
    print(f"  attn[0]: {t_out['attn_weights'][0].shape}  (B, h, T, T)")

    print(f"\nOscillator forward pass:")
    print(f"  logits : {osc_out['logits'].shape}")
    print(f"  gate[0]: {osc_out['gates'][0].shape}  (B, h, T, d_k)")
    print(f"  λ per layer: {[f'{v:.3f}' for v in osc_out['lam_values']]}")

    # ── 坍缩门控可视化（第 0 层，第 0 头，前 5 token）──────────
    gate0 = osc_out['gates'][0][0, 0, :5, 0]   # (5,)
    print(f"\n  Layer-0 Head-0 collapse gate (token 0-4, dim 0):")
    print(f"  {gate0.detach().tolist()}")
    print(f"  (emerged from phase propagation, not softmax(QK^T))")

    # ── 架构核心差异 ──────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Core difference:")
    print("  Transformer:  attn = softmax(QK^T / sqrt(d)) * V")
    print("                <- similarity computed explicitly, one shot")
    print()
    print("  Oscillator:   D_i(t+1) = (1-lam)*D_i + lam*sum(A_ij*D_j)")
    print(f"                <- {osc_cfg.n_otal_steps} steps of phase propagation, attention emerges from dynamics")
    print(f"                   lam is learnable (init {osc_cfg.lam})")
    print()
    print("  OTAL (QCU fast-search) = OscillatorAttention prototype on a graph")
    print("  OscillatorAttention    = OTAL in differentiable matrix form")
    print("-" * 60)


if __name__ == "__main__":
    main()
