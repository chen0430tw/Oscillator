# transformer/example.py
"""
Quick sanity-check: build a small Transformer and run a forward pass.

Usage
-----
    python transformer/example.py
"""

import torch
from .model import Transformer, TransformerConfig


def main():
    # ---- tiny model (fits on CPU for quick check) ----
    cfg = TransformerConfig(
        vocab_size  = 1000,
        max_seq_len = 64,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 2,
        d_ff        = 512,
        dropout     = 0.0,
    )

    model = Transformer(cfg)
    print(f"Transformer — {model.param_count()} parameters")
    print(f"  d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"d_k={cfg.d_k}, n_layers={cfg.n_layers}")

    # ---- forward pass ----
    B, T = 2, 16
    tokens = torch.randint(1, cfg.vocab_size, (B, T))   # avoid pad_id=0
    tokens[0, -3:] = 0                                   # add some padding

    out = model(tokens, causal=False)
    print(f"\nForward pass (causal=False):")
    print(f"  input  : {tokens.shape}")
    print(f"  logits : {out['logits'].shape}")
    print(f"  hidden : {out['hidden'].shape}")
    print(f"  attn[0]: {out['attn_weights'][0].shape}  (B, h, T, T)")

    # ---- causal mode ----
    out_causal = model(tokens, causal=True)
    print(f"\nForward pass (causal=True):")
    print(f"  logits : {out_causal['logits'].shape}")

    # ---- compare with Oscillator ----
    print("\n" + "─" * 60)
    print("Architecture comparison:")
    print("  Transformer:  score(Q,K) = softmax(QKᵀ / √d_k) · V")
    print("                             ← similarity computed explicitly")
    print("  Oscillator:   dρ/dt = −i[H,ρ] + Σ D[c_k]ρ")
    print("                             ← similarity emerges from dynamics")
    print("─" * 60)


if __name__ == "__main__":
    main()
