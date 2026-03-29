# -*- coding: utf-8 -*-
from __future__ import annotations
"""
train_talk.py — 用 HLBD 中文句子训练"会说话"的 Oscillator

只取 HLBD 里的中文部分：
  concept（词卡）+ level_2（短语）+ level_6（中文句子）

去掉多语言结构符号，让模型学纯中文。
"""

import sys, io, json, math, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/Oscillator")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from oscillator.model import Oscillator
from oscillator.config import OscillatorConfig

HLBD_PATH = "D:/APT-Transformer/data/HLBD_Full_V2.json"

# ── 1. 只提取中文文本 ──────────────────────────────────────────────

def load_chinese(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for s in data["samples"]:
        # 只取中文例句，避免字卡/短语拼接导致重复
        if "level_6" in s and "中文" in s["level_6"]:
            sent = s["level_6"]["中文"].strip()
            if sent:
                texts.append(sent)
    return texts


# ── 2. 字符级 Tokenizer ────────────────────────────────────────────

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


# ── 3. Dataset ─────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts: list[str], tok: CharTokenizer, seq_len: int):
        all_ids = []
        for t in texts:
            all_ids.extend(tok.encode(t))
            all_ids.append(tok.pad_id)
        self.ids     = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ── 4. 训练 ────────────────────────────────────────────────────────

def train(model, loader, n_steps, lr, device, label):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10, final_div_factor=10,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses, step = [], 0
    t0 = time.perf_counter()
    while step < n_steps:
        for x, y in loader:
            if step >= n_steps:
                break
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out["logits"].view(-1, out["logits"].size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                avg = sum(losses[-200:]) / 200
                print(f"  [{label}] step {step:>5}/{n_steps}  loss={avg:.4f}  "
                      f"{time.perf_counter()-t0:.1f}s")
    return losses


# ── 5. 生成（temperature + top-k 采样）────────────────────────────

def generate(model, tok, prefix: str, max_new: int, seq_len: int,
             device: str, temperature: float = 0.8, top_k: int = 40,
             rep_penalty: float = 1.8, rep_window: int = 16) -> str:
    model.eval()
    ids = torch.tensor([tok.encode(prefix)], dtype=torch.long, device=device)
    stop_id = tok._ch2id.get("。", None)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids[:, -seq_len:])["logits"][0, -1]  # (vocab,)
            # 重复惩罚：最近 rep_window 个 token 降低被选概率
            recent = ids[0, -rep_window:].tolist()
            for prev in set(recent):
                if prev > 1:  # 非 pad/unk
                    if logits[prev] > 0:
                        logits[prev] /= rep_penalty
                    else:
                        logits[prev] *= rep_penalty
            logits = logits / temperature
            # top-k 过滤
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[-1]] = float("-inf")
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tok.pad_id or next_id == stop_id:
                break
    return tok.decode(ids[0].tolist())


# ── 6. 主程序 ──────────────────────────────────────────────────────

def main():
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    SEQ_LEN = 96
    BATCH   = 64
    N_STEPS = 250
    LR      = 2e-3
    SEED    = 42

    torch.manual_seed(SEED)
    random.seed(SEED)

    print("加载 HLBD 中文部分...")
    texts = load_chinese(HLBD_PATH)
    tok   = CharTokenizer(texts)
    print(f"  样本数: {len(texts)}  词表大小: {tok.vocab_size}")
    print(f"  示例: {texts[0]}")
    print(f"  示例: {texts[1]}")

    n_train = int(len(texts) * 0.9)
    ds      = TextDataset(texts[:n_train], tok, SEQ_LEN)
    loader  = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    print(f"  训练字符数: {len(ds)}  batch数: {len(loader)}")

    # 较大的 Oscillator sparse 模型
    cfg = OscillatorConfig(
        vocab_size   = tok.vocab_size,
        max_seq_len  = SEQ_LEN,
        d_model      = 128,
        n_heads      = 4,
        n_layers     = 3,
        d_ff         = 512,
        dropout      = 0.15,
        n_otal_steps = 3,
        lam          = 0.3,
        phase_topk   = SEQ_LEN // 4,
        adj_topk     = SEQ_LEN // 4,
    )
    model = Oscillator(cfg)
    print(f"\nOscillator sparse  params: {model.param_count()}")
    print(f"设备: {DEVICE}")

    print(f"\n{'='*55}")
    print(f"训练 {N_STEPS} steps...")
    losses = train(model, loader, N_STEPS, LR, DEVICE, "Osc")

    final_loss = sum(losses[-200:]) / 200
    print(f"\n最终 loss（最后200步均值）: {final_loss:.4f}")

    # 生成示例
    print(f"\n{'='*55}")
    prefixes = ["视频", "今天", "我想", "学习", "在外"]
    temps    = [0.8, 1.0, 1.2]
    for prefix in prefixes:
        print(f"\n前缀「{prefix}」:")
        for t in temps:
            out = generate(model, tok, prefix, max_new=60, seq_len=SEQ_LEN,
                           device=DEVICE, temperature=t, top_k=40)
            print(f"  t={t}: {out}")


if __name__ == "__main__":
    main()
