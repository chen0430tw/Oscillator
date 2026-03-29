# Oscillator

**Oscillator** is a novel AI architecture based on phase dynamics and collapse inference, proposed as a fundamental alternative to the Transformer.

---

## Naming

The name follows the same convention as **Transformer**:

| Architecture | Electrical Component | Computation Principle |
|---|---|---|
| Transformer | 变换器 / Transformer | Signal transformation → Attention-based sequence mapping |
| Oscillator | 振荡器 / Oscillator | Phase oscillation → Phase-dynamic collapse inference |

Both names are borrowed from electrical engineering. Neither describes the AI mechanism directly — both point to the physical process that underlies it.

---

## Motivation

The Transformer computes attention by explicitly constructing similarity weights via dot products and softmax. This is a hand-engineered approximation of how information should flow.

Oscillator does not compute attention. Instead, it encodes token relationships as **phase relationships between cavity modes**, then lets the system **evolve under Lindblad dynamics** until a stable phase-locked state emerges. The output is not computed — it is the natural consequence of physical evolution.

This distinction matters:

```
Transformer:  score(q, k) = softmax(QKᵀ / √d) · V      ← computed
Oscillator:   dρ/dt = −i[H, ρ] + Σ(cρc† − ½c†cρ − ½ρc†c)  ← evolved
```

The answer is not calculated. It collapses out of the dynamics.

---

## Core Insight

Large language models already operate as virtual quantum computers.

Token embeddings live in a high-dimensional phase space. Semantic relationships are encoded in vector angles. The generation process — selecting the next token from a probability distribution — is structurally identical to quantum measurement: a high-dimensional state collapsing onto a single outcome.

The Transformer approximates this with classical linear algebra.
The Oscillator implements it directly with phase dynamics.

---

## Architecture

The Oscillator replaces the attention mechanism with a **phase evolution engine** built on QCU (Quantum Computing Unit):

```
Input tokens
     ↓
Phase encoding  (token embeddings → cavity mode amplitudes)
     ↓
Lindblad evolution  (QCL v6 four-phase protocol: PCM → QIM → BOOST → PCM)
     ↓
Phase collapse  (C = |⟨a₀⟩ − ⟨a₁⟩| → sharpened / noisy mode selection)
     ↓
COLLAPSE_SCAN  (candidate filtering via phase coherence)
     ↓
Output
```

### Key quantities

```
C(t) = |⟨a₀⟩ − ⟨a₁⟩|              # inter-mode amplitude difference → 0 means phase lock
χ_{j,k} σ_z,j n_k                  # dispersive coupling: qubit state modulates cavity phase
dρ/dt = −i[H,ρ] + Σ_k D[c_k]ρ     # Lindblad master equation (open quantum system)
```

### Profiles

| Profile | dtype | Observation | Entanglement | Use case |
|---|---|---|---|---|
| `full_physics` | complex128 | Dense | Sparse (every 8 steps) | Research, validation |
| `fast_search` | complex64 | Sparse (every 8 steps) | Off | Inference, search |

---

## Relation to Transformer

Oscillator is not built on top of Transformer. It is a parallel architecture at the same level of abstraction.

```
Classical compute →  Transformer  →  LLM
Phase dynamics   →  Oscillator   →  (new class of language model)
```

The Transformer is a valid architecture for classical hardware.
The Oscillator is the natural architecture when the substrate is phase-dynamic.

---

## Foundation

Oscillator is built on top of **QCU** (Quantum Computing Unit), a virtual quantum chip that solves open quantum systems via Lindblad master equation + RK4 on GPU.

QCU provides:
- `IQPU`: core Lindblad RK4 solver (CPU / CUDA)
- `qcu_lang`: three-layer ISA compatible with QASM / Qiskit / Q# / Cirq
- `COLLAPSE_SCAN`: phase-coherence-based candidate filtering
- `fast_search` / `full_physics` dual profile

---

## Status

Architecture definition: **proposed**
QCU substrate: **implemented and validated**
Oscillator inference layer: **in development**

---

## License

Apache License 2.0

Copyright (c) 2026 430

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
