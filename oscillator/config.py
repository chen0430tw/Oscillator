# oscillator/config.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OscillatorConfig:
    # ── Token / sequence ──────────────────────────────────────
    vocab_size:   int   = 32000
    max_seq_len:  int   = 512
    d_model:      int   = 512
    n_heads:      int   = 8
    n_layers:     int   = 6
    d_ff:         int   = 2048
    dropout:      float = 0.1
    pad_id:       int   = 0

    # ── Oscillator-specific ───────────────────────────────────
    # n_otal_steps: propagation steps (depth of phase evolution)
    # Transformer analogue: none — this has no equivalent.
    # More steps → slower but deeper phase synchronization.
    n_otal_steps: int   = 3

    # lam: neighborhood coupling coefficient (λ in OTAL update rule)
    # D_i(t+Δt) = (1-λ)·D_i + λ·Σ Ã_ij·D_j
    # λ→0: each token ignores neighbors (identity)
    # λ→1: token fully replaced by neighbor average
    lam:          float = 0.3

    # collapse_mode: how to convert complex D → real attention gate
    # "phase_align" : alignment with mean-field direction (Kuramoto-inspired)
    # "amplitude"   : |D|² normalized across tokens
    collapse_mode: str  = "phase_align"

    # phase_topk: sparse phase attention — only keep top-k aligned token pairs
    # None → dense (full T×T); int → sparse O(T·k) memory
    # Quantum analogue: partial collapse — only the k strongest-coupled
    # oscillator pairs participate in phase synchronization.
    phase_topk: int | None = None

    # adj_topk: sparse initial adjacency Ã — top-k neighbors per token
    # None → dense softmax; int → sparse O(T·k)
    adj_topk:   int | None = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.d_k = self.d_model // self.n_heads
