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

## 2026-03-31 Update

- Real training bug found and fixed:
  - `train_talk.py` and `train_hlbd.py` were packing samples with `pad_id=0` as the stream separator
  - this can create fully masked rows under `pad_mask + causal_mask`
  - symptom: first-batch `NaN`
  - fix: introduce `sep_id` and keep `pad_id` only for real padding
- After the separator fix, Nano5 smoke tests confirmed:
  - finite logits
  - finite gradients
  - normal loss descent
- 1000-step real-data comparison on `HLBD_Full_V2.json` now shows:
  - `learned`: val `0.8935`
  - `local_window`: val `0.8881`
  - `blind_uniform` (distance-biased, content-blind): val `0.8966`
  - `identity`: val `0.9450`
- Strong interpretation:
  - token-to-token propagation matters
  - pure learned graph is not the best current default
  - fixed local topology is currently stronger than pure learned coupling
- New best hybrid result:
  - `mixed_local` reworked as `local prior logits + beta * learned residual logits`
  - beta sweep:
    - `beta=0.03`: val `0.8890`
    - `beta=0.10`: val `0.8909`
    - `beta=0.30`: val `0.8898`
  - this beats pure `learned`, but still does not beat pure `local_window`
- Current default engineering decision:
  - broad real-data sweep now favors `mixed_local` over pure `local_window`
  - best observed setting in the broad sweep: `adj_mode="mixed_local", adj_mix_beta=0.30`
  - training defaults should now use `mixed_local` with `beta=0.30`

## 2026-03-31 QuickCook Login-Node Update

- Dedicated login-node launcher now exists on Nano5:
  - `/work/twsuday816/APT-Transformer/slurm/run_osc_quickcook_login.sh`
- This script is the preferred entry point for short login-node `osc_quickcook` runs.
- Remote stability fixes already applied in `pretrain_quickcook.py`:
  - stream read exceptions are caught and skipped with warning instead of killing the run
  - DataLoader login-node settings use `num_workers=1` and `prefetch_factor=2`
- Wrapper / launcher notes:
  - timestamped output directories
  - configurable `COMPILE_THREADS` -> `TORCHINDUCTOR_COMPILE_THREADS`
  - verified on Nano5 login node with successful checkpoint/tokenizer save

### Login-Node Thread Sweep

All runs below completed `200` steps successfully with stable GPU memory.

- `COMPILE_THREADS=4`: `tok/s ~ 40,751`
- `COMPILE_THREADS=7`: `tok/s ~ 37,848`
- `COMPILE_THREADS=8`: `tok/s ~ 45,315`
- `COMPILE_THREADS=9`: `tok/s ~ 45,685`  <- current best
- `COMPILE_THREADS=10`: `tok/s ~ 38,577`
- `COMPILE_THREADS=11`: `tok/s ~ 37,426`
- `COMPILE_THREADS=12`: `tok/s ~ 44,820`
- `COMPILE_THREADS=16`: `tok/s ~ 42,990`

### Current Operational Conclusion

- Best verified login-node quickcook setting so far:
  - `COMPILE_THREADS=9`
- Near-best alternatives:
  - `8`
  - `12`
- Practical interpretation:
  - performance peak is in the `8-9-12` region
  - current sweet spot is `9`
- This result is about login-node quickcook throughput tuning, not the core Oscillator math.
- Keep the model-side default as:
  - `adj_mode="mixed_local"`
  - `adj_mix_beta=0.30`

## 2026-03-31 QuickCook Trajectory Update

- Verified login-node quickcook checkpoints now cover:
  - `500`
  - `1000`
  - `3000`
  - `5000`
  - `10000`
  - `20000`

### QuickCook Loss Curve

- `500`: `8.2152`
- `1000`: `7.8227`
- `3000`: `7.5492`
- `5000`: `6.8315`
- `10000`: `6.1685`
- `20000`: `5.5746`

### Qualitative Generation Stages

- `500 -> 3000`
  - still mostly token soup / punctuation fragments
  - no stable sentence structure yet
- `5000`
  - English side starts looking sentence-like
  - Chinese side still heavily corrupted / fragmented
- `10000`
  - enters a "half-formed Chinese" stage
  - short clause shapes appear, but many malformed characters remain
- `20000`
  - first checkpoint that clearly produces readable Chinese fragments
  - still noisy and not production-quality, but already beyond pure fragment flow

### Practical Interpretation

- QuickCook + HLBD is not stalled; the learning curve is real but front-loaded with noisy early behavior.
- `20000` is the first qualitative breakpoint worth highlighting in a paper-style trajectory figure.
- A useful qualitative narrative is:
  - fragment soup -> language silhouette -> half-formed Chinese -> readable Chinese fragments

## 2026-03-31 QuickCook Transformer Baseline Update

- A matched `Transformer` baseline was added to the Nano5 login-node quickcook launcher:
  - `bash slurm/run_osc_quickcook_login.sh --model-arch transformer --max-steps N`
- Verified baseline checkpoints now cover:
  - `500`
  - `1000`
  - `3000`
  - `5000`
  - `10000`
  - `20000`

### Oscillator vs Transformer Loss

- `500`
  - Oscillator: `8.2152`
  - Transformer: `8.2125`
- `1000`
  - Oscillator: `7.8227`
  - Transformer: `7.8180`
- `3000`
  - Oscillator: `7.5492`
  - Transformer: `7.5476`
- `5000`
  - Oscillator: `6.8315`
  - Transformer: `7.6165`
- `10000`
  - Oscillator: `6.1685`
  - Transformer: `5.4714`
- `20000`
  - Oscillator: `5.5746`
  - Transformer: `5.0540`

### Throughput / Memory

- At `20000` steps:
  - Oscillator: `148,184 tok/s`, `reserved 5.46GB`, `peak 4.48GB`
  - Transformer: `282,528 tok/s`, `reserved 4.81GB`, `peak 3.83GB`

### Qualitative Comparison

- `500 -> 3000`
  - both models are still mostly fragment streams
  - neither has stable Chinese sentence structure yet
- `5000`
  - Oscillator is briefly ahead on loss
  - Transformer text is still mostly mixed fragments
- `10000`
  - Oscillator: enters half-formed Chinese, still noisy
  - Transformer: already produces longer readable Chinese-like clauses and discourse markers
- `20000`
  - Oscillator: readable Chinese fragments, but still rough and unstable
  - Transformer: clearly more coherent and readable overall, though still far from production-quality generation

### Current Interpretation

- On this QuickCook + HLBD trajectory, Oscillator is competitive in the early phase and even briefly ahead at `5000`.
- By `10000 -> 20000`, Transformer converges faster, generates cleaner text, and is much more compute-efficient.
- Current practical baseline conclusion:
  - Transformer remains the stronger reference architecture on this task/setup
  - Oscillator still needs more architecture work to justify its extra compute cost

## 2026-03-31 Tensorearch Profiling v3 Update

- The first profiling traces were too coarse:
  - `Transformer` only had `attn / ffn`
  - `Oscillator` only had `phase / prop / ffn`
  - the graph was effectively a single chain
- That made the first Tensorearch result low-value:
  - `route_entropy -> 0`
  - `effect_entropy -> 0`
  - `intelligence -> 0`

### Profiling v3 Slice Split

- Transformer slices now export:
  - `q`
  - `k`
  - `v`
  - `score`
  - `attn_out`
  - `ffn`
- Oscillator slices now export:
  - `phase`
  - `adj`
  - `prop`
  - `phase_attn`
  - `value`
  - `attn_out`
  - `ffn`

### Profiling v3 Branch Structure

- Transformer trace now contains real branching:
  - `q -> score`
  - `k -> score`
  - `v -> attn_out`
  - `score -> attn_out`
  - `attn_out -> ffn`
  - `ffn / attn_out -> next q`
- Oscillator trace now contains real branching:
  - `phase -> adj`
  - `phase -> prop`
  - `adj -> prop`
  - `prop -> phase_attn`
  - `phase -> phase_attn`
  - `value / phase_attn -> attn_out`
  - `attn_out -> ffn`
  - `ffn / attn_out -> next phase`

### Short Profiling Compare (`30` steps, Nano5 login node)

- Transformer
  - bottleneck: `blk0.score`
  - obedience: `0.3496`
  - intelligence: `0.0557`
  - coupling: `0.0145`
  - throughput: `~9038 tok/s`
- Oscillator
  - bottleneck: `blk0.adj`
  - obedience: `0.3465`
  - intelligence: `0.0856`
  - coupling: `0.0156`
  - throughput: `~8514 tok/s`

### Practical Interpretation

- v3 profiling finally activates non-zero `intelligence`.
- Transformer's main local hotspot is now specifically the attention score path, not generic `attn`.
- Oscillator's main local hotspot is now specifically adjacency construction, not generic `phase/prop`.
- On this short profiling metric:
  - Oscillator shows higher structural intelligence than Transformer
  - but still lower throughput
- This is a profiling-structure result, not a long-run quality claim.
- Current architectural implication:
  - Transformer remains the stronger practical baseline on `QuickCook + HLBD`
  - Oscillator's next optimization target is `adj` construction cost and routing efficiency
