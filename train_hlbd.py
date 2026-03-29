# -*- coding: utf-8 -*-
from __future__ import annotations
"""
train_hlbd.py — 用 HLBD 数据训练 Oscillator vs Transformer

用法：
    cd D:/Oscillator
    py -3.13 train_hlbd.py
"""

import sys, io, json, math, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from oscillator.model import Oscillator
from oscillator.config import OscillatorConfig
from transformer.model import Transformer, TransformerConfig


# ── 1. 数据加载 ────────────────────────────────────────────────────────

HLBD_PATH = "D:/APT-Transformer/data/HLBD_Full_V2.json"

def load_hlbd(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    def flatten(s):
        parts = [s["concept"]]
        for k in ["level_1","level_2","level_3","level_4",
                  "level_5","level_6","level_7","level_8"]:
            parts.extend(str(v) for v in s[k].values())
        return " | ".join(parts)

    return [flatten(s) for s in samples]


# ── 2. 字符级 Tokenizer ────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, texts: list[str]):
        chars = sorted(set("".join(texts)))
        self.pad_id  = 0
        self.unk_id  = 1
        self._ch2id  = {c: i+2 for i, c in enumerate(chars)}
        self._id2ch  = {i+2: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 2

    def encode(self, text: str) -> list[int]:
        return [self._ch2id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2ch.get(i, "?") for i in ids if i > 1)


# ── 3. Dataset ─────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tok: CharTokenizer, seq_len: int):
        # 把所有文本连接成一个大 token 序列
        all_ids = []
        for t in texts:
            all_ids.extend(tok.encode(t))
            all_ids.append(tok.pad_id)   # 样本间分隔
        self.ids     = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ── 4. 训练函数 ────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    loader: DataLoader,
    n_steps: int,
    lr: float = 1e-3,
    device: str = "cpu",
    label: str = "",
) -> list[float]:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []
    step   = 0
    t0     = time.perf_counter()

    while step < n_steps:
        for x, y in loader:
            if step >= n_steps:
                break
            x, y = x.to(device), y.to(device)

            out    = model(x)
            logits = out["logits"]                    # (B, T, vocab)
            loss   = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1

            if step % 50 == 0:
                avg = sum(losses[-50:]) / 50
                elapsed = time.perf_counter() - t0
                print(f"  [{label}] step {step:>4}/{n_steps}  "
                      f"loss={avg:.4f}  {elapsed:.1f}s")

    return losses


# ── 5. 主程序 ──────────────────────────────────────────────────────────

def main():
    DEVICE   = "cpu"
    SEQ_LEN  = 48
    BATCH    = 32
    N_STEPS  = 300
    LR       = 1e-3
    SEED     = 42

    torch.manual_seed(SEED)
    random.seed(SEED)

    # 数据
    print("加载 HLBD...")
    texts = load_hlbd(HLBD_PATH)
    tok   = CharTokenizer(texts)
    print(f"  样本数: {len(texts)}  词表大小: {tok.vocab_size}")

    # 取前 80% 做训练
    n_train = int(len(texts) * 0.8)
    ds    = TextDataset(texts[:n_train], tok, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    print(f"  训练序列数: {len(ds)}  batch 数/epoch: {len(loader)}")

    # 模型配置（相同规模）
    common = dict(
        vocab_size  = tok.vocab_size,
        max_seq_len = SEQ_LEN,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 2,
        d_ff        = 512,
        dropout     = 0.1,
    )
    t_cfg   = TransformerConfig(**common)
    osc_cfg = OscillatorConfig(**common, n_otal_steps=3, lam=0.3)

    transformer = Transformer(t_cfg)
    oscillator  = Oscillator(osc_cfg)

    print(f"\nTransformer  params: {transformer.param_count()}")
    print(f"Oscillator   params: {oscillator.param_count()}")

    # 训练
    print(f"\n{'='*55}")
    print(f"训练 Transformer ({N_STEPS} steps)...")
    t_losses = train(transformer, loader, N_STEPS, lr=LR,
                     device=DEVICE, label="Transformer")

    print(f"\n训练 Oscillator ({N_STEPS} steps)...")
    osc_losses = train(oscillator, loader, N_STEPS, lr=LR,
                       device=DEVICE, label="Oscillator")

    # 结果对比
    def smooth(ls, w=20):
        return [sum(ls[max(0,i-w):i+1])/min(i+1,w) for i in range(len(ls))]

    t_smooth   = smooth(t_losses)
    osc_smooth = smooth(osc_losses)

    print(f"\n{'='*55}")
    print(f"最终 loss 对比（最后 50 步均值）：")
    print(f"  Transformer : {sum(t_losses[-50:])/50:.4f}")
    print(f"  Oscillator  : {sum(osc_losses[-50:])/50:.4f}")

    # Loss 曲线（文字版，每 50 步一个点）
    print(f"\nLoss 曲线（每 50 步）：")
    print(f"  {'step':>5}  {'Transformer':>12}  {'Oscillator':>12}")
    for i in range(0, N_STEPS, 50):
        t_avg   = sum(t_losses[i:i+50]) / min(50, len(t_losses[i:i+50]))
        osc_avg = sum(osc_losses[i:i+50]) / min(50, len(osc_losses[i:i+50]))
        bar_t   = "█" * int(t_avg * 5)
        bar_o   = "█" * int(osc_avg * 5)
        print(f"  {i+50:>5}  {t_avg:>12.4f}  {osc_avg:>12.4f}")

    # 生成示例
    print(f"\n{'='*55}")
    print("生成示例（贪心解码，前缀='视频'）：")
    for name, model in [("Transformer", transformer), ("Oscillator", oscillator)]:
        model.eval()
        prefix = tok.encode("视频")
        ids = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            for _ in range(30):
                out = model(ids[:, -SEQ_LEN:])
                next_id = out["logits"][0, -1].argmax().item()
                ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        generated = tok.decode(ids[0].tolist())
        print(f"  {name}: {generated[:60]}")


if __name__ == "__main__":
    main()
