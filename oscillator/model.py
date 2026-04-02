# oscillator/model.py
"""
Oscillator — 振荡器语言模型，兼容 HuggingFace PreTrainedModel。

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
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

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

class Oscillator(PreTrainedModel, GenerationMixin):
    """振荡器语言模型，基于相位动力学的 Transformer 替代架构。

    兼容 HuggingFace PreTrainedModel：
      - 支持 model.save_pretrained() / from_pretrained()
      - 支持 model.generate()（继承自 GenerationMixin）
      - forward() 返回 CausalLMOutputWithPast，兼容 output["logits"]

    Parameters
    ----------
    config : OscillatorConfig
    """

    config_class = OscillatorConfig
    _tied_weights_keys = ["lm_head.weight"]  # tied to embed.weight

    def __init__(self, config: OscillatorConfig):
        super().__init__(config)
        self.cfg = config   # 保留 self.cfg 供现有代码访问

        self.embed   = nn.Embedding(config.vocab_size, config.d_model,
                                    padding_idx=config.pad_id)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_len,
                                          config.dropout)
        self.layers  = nn.ModuleList(
            [OscillatorBlock(config) for _ in range(config.n_layers)]
        )
        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tie embeddings

        self.post_init()   # HF 标准初始化钩子

    def _init_weights(self, module):
        """HF PreTrainedModel 要求实现此方法。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if module.weight.dim() > 1:
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # HF tied-weight hooks — needed for save/load to correctly re-tie lm_head ↔ embed
    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def make_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()\
                     .unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        causal:         bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Parameters
        ----------
        input_ids      : (B, T)
        attention_mask : (B, T)  — 1 = keep, 0 = mask（HF 标准；内部转为 bool mask）
        labels         : (B, T)  — -100 表示忽略位置
        causal         : bool    — 是否使用因果掩码（默认 True，用于自回归生成）

        Returns
        -------
        CausalLMOutputWithPast
            .loss   : scalar（仅当传入 labels 时）
            .logits : (B, T, vocab_size)
        """
        B, T = input_ids.shape

        # 构建掩码
        pad_mask = self.make_pad_mask(input_ids)
        if causal:
            mask = pad_mask | self.make_causal_mask(T, input_ids.device)
        else:
            mask = pad_mask

        # 如果上层传了 attention_mask（HF 格式 1/0），合并进来
        if attention_mask is not None:
            hf_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            mask = mask | hf_mask

        x = self.pos_enc(self.embed(input_ids) * math.sqrt(self.cfg.d_model))

        gates      = []
        lam_values = []
        for layer in self.layers:
            x, gate = layer(x, mask)
            gates.append(gate)
            # .item() removed: causes graph break under torch.compile
            lam_values.append(
                torch.sigmoid(layer.attn.propagate.logit_lam).detach()
            )

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        """HF generate() 每步调用此方法准备输入。"""
        return {"input_ids": input_ids}

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        if n >= 1_000_000_000:
            return f"{n/1e9:.2f}B"
        elif n >= 1_000_000:
            return f"{n/1e6:.1f}M"
        return f"{n/1e3:.1f}K"
