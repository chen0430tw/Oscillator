# oscillator/config.py
from __future__ import annotations
from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig


class OscillatorConfig(PreTrainedConfig):
    """Oscillator 模型配置，兼容 HuggingFace PreTrainedConfig。"""

    model_type = "oscillator"

    def __init__(
        self,
        # ── Token / sequence ──────────────────────────────────────
        vocab_size:    int   = 32000,
        max_seq_len:   int   = 512,
        d_model:       int   = 512,
        n_heads:       int   = 8,
        n_layers:      int   = 6,
        d_ff:          int   = 2048,
        dropout:       float = 0.1,
        pad_id:        int   = 0,
        bos_token_id:  int   = 1,
        eos_token_id:  int   = 2,
        # ── Oscillator-specific ───────────────────────────────────
        n_otal_steps:  int   = 3,
        lam:           float = 0.3,
        collapse_mode: str   = "phase_align",
        phase_topk:    int | None = None,
        adj_topk:      int | None = None,
        **kwargs,
    ):
        # pop HF token ids from kwargs to avoid duplicate keyword arg on from_pretrained
        _pad = kwargs.pop("pad_token_id", pad_id)
        _bos = kwargs.pop("bos_token_id", bos_token_id)
        _eos = kwargs.pop("eos_token_id", eos_token_id)
        super().__init__(
            pad_token_id=_pad,
            bos_token_id=_bos,
            eos_token_id=_eos,
            **kwargs,
        )
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.vocab_size    = vocab_size
        self.max_seq_len   = max_seq_len
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.n_layers      = n_layers
        self.d_ff          = d_ff
        self.dropout       = dropout
        self.pad_id        = pad_id
        self.n_otal_steps  = n_otal_steps
        self.lam           = lam
        self.collapse_mode = collapse_mode
        self.phase_topk    = phase_topk
        self.adj_topk      = adj_topk
        self.d_k           = d_model // n_heads
        self.num_hidden_layers = n_layers   # required by HF DynamicCache in generate()
