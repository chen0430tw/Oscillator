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
        self.sep_id  = 2
        self._ch2id  = {c: i+3 for i, c in enumerate(chars)}
        self._id2ch  = {i+3: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 3

    def encode(self, text: str) -> list[int]:
        return [self._ch2id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2ch.get(i, "?") for i in ids if i > 2)


# ── 3. Dataset ─────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tok: CharTokenizer, seq_len: int):
        # 把所有文本连接成一个大 token 序列
        all_ids = []
        for t in texts:
            all_ids.extend(tok.encode(t))
            # Use a non-pad separator in the packed token stream.
            all_ids.append(tok.sep_id)
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # 余弦退火：从 lr 线性 warmup 到 lr，再余弦退火到 lr/10
    warmup_steps = max(1, n_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=10,
    )
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
            scheduler.step()

            losses.append(loss.item())
            step += 1

            if step % 50 == 0:
                avg = sum(losses[-50:]) / 50
                elapsed = time.perf_counter() - t0
                # 收集 Oscillator 惯性统计
                active_info = ""
                for mod in model.modules():
                    from oscillator.attention import TopologyPropagation
                    if isinstance(mod, TopologyPropagation) and mod.last_active_ratios:
                        ratios = mod.last_active_ratios
                        active_info = f"  active={[f'{r:.0%}' for r in ratios]}"
                        break
                print(f"  [{label}] step {step:>4}/{n_steps}  "
                      f"loss={avg:.4f}  {elapsed:.1f}s{active_info}")

    return losses


# ── 5. 主程序 ──────────────────────────────────────────────────────────

def main():
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN  = 48
    BATCH    = 64    # GPU 上加大 batch
    N_STEPS  = 600
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
    # top-k = T//4 = 12，即每个 token 只和最对齐的 12 个 token 同步
    TOPK = SEQ_LEN // 4
    t_cfg        = TransformerConfig(**common)
    osc_cfg      = OscillatorConfig(
        **common,
        n_otal_steps=3,
        lam=0.3,
        adj_mode="mixed_local",
        adj_mix_beta=0.30,
    )
    osc_sp_cfg   = OscillatorConfig(
        **common,
        n_otal_steps=3,
        lam=0.3,
        adj_mode="mixed_local",
        adj_mix_beta=0.30,
        phase_topk=TOPK,
        adj_topk=TOPK,
    )

    transformer   = Transformer(t_cfg)
    oscillator    = Oscillator(osc_cfg)
    oscillator_sp = Oscillator(osc_sp_cfg)

    print(f"\nTransformer       params: {transformer.param_count()}")
    print(f"Oscillator(dense) params: {oscillator.param_count()}")
    print(f"Oscillator(sparse k={TOPK}) params: {oscillator_sp.param_count()}")

    # 训练
    print(f"\n{'='*55}")
    print(f"训练 Transformer ({N_STEPS} steps)...")
    t_losses = train(transformer, loader, N_STEPS, lr=LR,
                     device=DEVICE, label="Transformer  ")

    print(f"\n训练 Oscillator dense ({N_STEPS} steps)...")
    osc_losses = train(oscillator, loader, N_STEPS, lr=LR,
                       device=DEVICE, label="Osc-Dense    ")

    print(f"\n训练 Oscillator sparse k={TOPK} ({N_STEPS} steps)...")
    osc_sp_losses = train(oscillator_sp, loader, N_STEPS, lr=LR,
                          device=DEVICE, label=f"Osc-Sparse-{TOPK}")

    # 结果对比
    print(f"\n{'='*55}")
    print(f"最终 loss 对比（最后 50 步均值）：")
    print(f"  Transformer      : {sum(t_losses[-50:])/50:.4f}")
    print(f"  Oscillator dense : {sum(osc_losses[-50:])/50:.4f}")
    print(f"  Oscillator sparse: {sum(osc_sp_losses[-50:])/50:.4f}")

    # Loss 曲线（文字版，每 50 步一个点）
    print(f"\nLoss 曲线（每 50 步）：")
    print(f"  {'step':>5}  {'Transformer':>12}  {'Osc-Dense':>12}  {'Osc-Sparse':>12}")
    for i in range(0, N_STEPS, 50):
        def avg(ls): return sum(ls[i:i+50]) / min(50, len(ls[i:i+50]))
        print(f"  {i+50:>5}  {avg(t_losses):>12.4f}  {avg(osc_losses):>12.4f}  {avg(osc_sp_losses):>12.4f}")

    # 生成示例
    print(f"\n{'='*55}")
    print("生成示例（贪心解码，前缀='视频'）：")
    for name, model in [("Transformer  ", transformer),
                        ("Osc-Dense    ", oscillator),
                        (f"Osc-Sparse-{TOPK}", oscillator_sp)]:
        model.eval()
        prefix = tok.encode("视频")
        ids = torch.tensor([prefix], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            for _ in range(30):
                out = model(ids[:, -SEQ_LEN:])
                next_id = out["logits"][0, -1].argmax().item()
                ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        generated = tok.decode(ids[0].tolist())
        print(f"  {name}: {generated[:60]}")


if __name__ == "__main__":
    main()
