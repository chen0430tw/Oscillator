# Oscillator Collaboration Status

## Purpose

This file is a shared status bulletin for parallel agents working on `D:\Oscillator`.
Read this first before making changes.

It exists to prevent:
- repeating already-settled diagnosis work
- re-running the wrong remote code version
- treating the current failure as an environment problem
- regressing the fixed `phase_sim collapse` bug

## Project Context

- `Oscillator` is a language-model architecture built on ideas derived from `QCU`.
- `QCU` is the lower-level virtual quantum chip substrate:
  - Lindblad master equation
  - RK4 evolution
  - phase / collapse / readout runtime
  - CPU/CUDA profiles such as `fast_search` and `full_physics`
- `Oscillator` is not directly calling IQPU step-by-step inside the LM.
- Instead, it turns the QCU / OTAL phase-dynamics idea into a differentiable attention replacement.

## Core Equations

Per OscillatorAttention layer:

```math
D = W_{re}x + iW_{im}x
```

```math
\tilde{A}_{ij} = \mathrm{softmax}_j\left(\frac{Q_iK_j^T}{\sqrt{d_k}}\right)
```

```math
D^{(t+1)} = (1-\lambda)D^{(t)} + \lambda \tilde{A}D^{(t)}
```

```math
D' = D^{(n)} + \alpha D^{(0)}
```

```math
D_{\mathrm{centered}} = D' - \frac{1}{T}\sum_j D'_j
```

```math
D_{\mathrm{unit}} = \frac{D_{\mathrm{centered}}}{|D_{\mathrm{centered}}|}
```

```math
\mathrm{phase\_sim}_{ij} = \mathrm{Re}\left(D_{\mathrm{unit},i}\, \overline{D_{\mathrm{unit},j}}^{\,T}\right)\sqrt{2d_k}
```

```math
\mathrm{out} = \mathrm{softmax}(\mathrm{phase\_sim})V
```

## Current Engineering State

- Local source of truth:
  - `D:\Oscillator`
- Remote training copy on Nano5:
  - `/work/twsuday816/Oscillator`
- Verified by SHA256:
  - `README.md`
  - `train_hlbd.py`
  - `train_talk.py`
  - `oscillator/config.py`
  - `oscillator/model.py`
  - `oscillator/attention.py`
- Conclusion:
  - remote Nano5 code matches local latest code for the key Oscillator files

## Confirmed Findings

### 1. Fixed bug: `phase_sim collapse`

Original failure:

```math
D_i^{(n)} \approx \bar{D} = \frac{1}{T}\sum_j D_j^{(0)}
```

This made pairwise phase similarity nearly constant, so attention became almost uniform.

Applied fix:

```math
D_{\mathrm{centered}} = D' - \frac{1}{T}\sum_j D'_j
```

Then normalize and compute phase similarity from the centered state.

Interpretation:
- this bug is considered understood and fixed
- do not revert to absolute-phase similarity

### 2. Main unresolved bug: adjacency does not learn

Observed diagnosis:
- `A_entropy` stays almost unchanged through training
- `W_Q` and `W_K` appear to receive near-zero effective learning signal
- the propagation subgraph is close to gradient-dead

Operational conclusion:

```math
\frac{\partial \mathcal{L}}{\partial W_Q} \approx 0
```

This means the current model likely degenerates into a weak generator where:
- `phase_sim collapse` is no longer the main blocker
- but `\tilde{A}` is still not meaningfully learned
- learning pressure mostly falls onto the easier residual / value-side path

## Remote Training Status

On Nano5:
- `osc_talk` training has completed full runs up to `5000` steps
- checkpoints exist at:
  - `checkpoint_step_1000.pt`
  - `checkpoint_step_2000.pt`
  - `checkpoint_step_3000.pt`
  - `checkpoint_step_4000.pt`
  - `checkpoint_step_5000.pt`
- latest verified progress log tail:
  - final loss around `7.75`
- latest short inference test with `checkpoint_step_5000.pt`:
  - model loads
  - CUDA inference works
  - output is still fragmented / unusable for dialogue

Interpretation:
- this is not primarily an environment failure
- this is not primarily a version mismatch
- this is not primarily a checkpoint loading failure
- this is a model-learning failure

## Important Scripts

Local:
- `train_hlbd.py`
- `train_talk.py`

Remote useful scripts on Nano5:
- `/work/twsuday816/chat_oscillator.py`
- `/work/twsuday816/chat_osc_quickcook.py`
- `/work/twsuday816/debug_oscillator.py`

Important note:
- `chat_oscillator.py` was pointing at `checkpoint_step_3000.pt`
- latest test quality should be judged against `checkpoint_step_5000.pt`, not `3000`

## Do Not Repeat

- Do not spend time re-proving that local and remote key Oscillator files differ.
  - They already match.
- Do not misclassify the current issue as "Nano5 environment broken".
- Do not reopen the already-fixed `phase_sim collapse` as if it were still the primary blocker.
- Do not evaluate dialogue quality only on the `3000`-step checkpoint if `5000` is available.

## Session Log

### 2026-03-31 (Claude Sonnet 4.6)

**HF format conversion** — completed, committed `6a854e7`→`2f37eed`
- `OscillatorConfig` → `PreTrainedConfig`，加 `num_hidden_layers`
- `Oscillator` → `PreTrainedModel + GenerationMixin`，加 `get/set_input/output_embeddings`
- `forward()` 返回 `CausalLMOutputWithPast`，支持 `save_pretrained / from_pretrained / generate`
- 本地验证全通过（forward / generate / save / load / weights match）

**phase_sim collapse 修复** — committed `e732a56`
- 原因：传播后全局均值场同步，绝对相位无区分性
- 修复：mean-field subtraction（见 Core Equations）
- `log_alpha` init `-3 → 0`（D_0 残差 0.05 → 0.5）
- 修复后 phase\_sim\_mean `0.887 → 0.001`，std `0.06 → 0.33`

**训练结果**
- 5000 steps，loss 仍卡在 7.75，与修复前完全相同
- A\_entropy 全程 2.06 不变 → W\_Q/W\_K 梯度死，传播子图无效
- 确认：learning failure，非 environment failure

---

## Recommended Next Step

Priority order:

1. Gradient diagnosis on the propagation path.
   - inspect gradients for `W_Q`, `W_K`, `logit_lam`, and phase encoding weights
   - locate where the useful signal goes to zero
2. Only after that, consider replacing learned `\tilde{A}` with a fixed topology prior such as a local window graph.

Reason:
- changing `\tilde{A}` too early may hide the real failure mode
- gradient-path diagnosis is currently the highest-value next move
