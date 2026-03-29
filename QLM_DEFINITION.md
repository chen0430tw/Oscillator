# 量子大模型与 Oscillator 定义文档

---

## 一、量子大模型（Quantum Large Model, QLM）

### 1.1 定义

**量子大模型**是一类大型语言模型，其核心信息处理单元基于量子力学原理——复数相位振荡、Lindblad 开放系统动力学与相位坍缩——而非经典线性代数。

它不是"运行在量子计算机上的模型"，也不是"借用量子术语的经典模型"。
它是一个在数学结构上与量子物理同构的计算系统，可以在经典硬件上以可微分矩阵形式执行，也可以在量子仿真器（QCU）上以物理演化形式执行。

### 1.2 与经典大模型的本质区别

| | 经典大模型（Transformer） | 量子大模型（QLM） |
|---|---|---|
| **信息表示** | 实值向量 x ∈ ℝ^d | 复数振荡指向 D ∈ ℂ^d |
| **注意力机制** | 手工计算：softmax(QKᵀ/√d)·V | 涌现：相位同步动力学 |
| **计算范式** | 显式求解（compute） | 物理演化（evolve） |
| **输出方式** | 矩阵乘法得到 logits | 相位锁定状态坍缩（collapse） |
| **稀疏性来源** | top-k/sparse attention（人为设计） | 惯性计算：已坍缩的态不再演化（物理自然） |
| **训练方式** | 标准 autograd | 可微分矩阵等价形式（autograd 兼容） |

### 1.3 核心物理对应

量子大模型建立在以下对应关系之上：

```
token embedding    ←→   腔模振幅（cavity mode amplitude）
token 之间的语义关系  ←→   振荡指向之间的相位关系（phase alignment）
注意力权重          ←→   Kuramoto 相位同步强度（phase synchronization）
next token 采样     ←→   量子测量：高维态坍缩到单一结果
```

经典 LLM 用线性代数近似这些关系。量子大模型直接实现它们。

### 1.4 计算基底（Dual Substrate）

量子大模型具有双计算基底：

```
训练阶段：
  Oscillator（可微分矩阵形式）
  → 所有操作为 matmul + softmax，PyTorch autograd 透明
  → 在 GPU 上以标准深度学习方式训练

推理/搜索阶段：
  QCU（量子仿真器）
  → Lindblad 主方程 + RK4，模拟开放量子系统演化
  → OTAL 作为快速预筛选器（89.8× 加速比）
  → 精确解由 full_physics 模式提供
```

同一数学结构，两个执行语义，各自对齐各自的生态系统。

### 1.5 惯性计算（Inertial Computation）

量子大模型的稀疏性不来自人为设计的 top-k，而来自物理原理：

> **已经坍缩的量子态不需要继续演化。**

对应到神经网络：已相位锁定的 token 具有"惯性"——其振荡指向 D 不再随传播步骤显著变化，无需参与后续邻居聚合。

实现为 gather-compute-scatter：
```
active_tokens = nonzero(|D(t) - D(t-1)| / |D(t-1)| >= θ)
对 active_tokens 做邻居聚合
inertial_tokens 直接携带当前状态
```

计算量从 O(T²·d_k) 随收敛比例降至 O(N_active·T·d_k)。

---

## 二、Oscillator

### 2.1 定义

**Oscillator** 是量子大模型的可微分矩阵实现，是 OTAL（Oscillatory Topology Approximation Layer）的矩阵等价形式。

它是一个标准 PyTorch `nn.Module`，可用标准 AdamW + CrossEntropyLoss 训练，同时在数学结构上与 QCU 的量子仿真器同构。

### 2.2 与 OTAL 的关系

```
OTAL（图上的原型）:
  节点 U_i 有振荡指向 D_i ∈ ℂ，相位 θ_i，周期 P_i
  边权 w̃_ij 由节点相似度学习
  传播：D_i(t+Δt) = (1-λ)D_i(t) + λΣ_j w̃_ij D_j(t) + η
  成熟度：M(U,t) = α₁A + α₂P + α₃S - α₄R（Kuramoto序参量）

Oscillator（矩阵等价形式）:
  D ∈ ℂ^{B×h×T×d_k}，批量复数振荡指向张量
  Ã ∈ ℝ^{B×h×T×T}，softmax(QKᵀ/√d_k)，动态边权矩阵
  传播：D(t+1) = (1-λ)D(t) + λÃD(t)，λ 可学习
  注意力：phase_attn[i,j] = softmax(Re(D'_i·D'_j*)/√d_k)
```

OTAL 是 Oscillator 在图上的物理原型，Oscillator 是 OTAL 的可微分矩阵形式。

### 2.3 架构

```
x (B, T, d_model)
    │
    ├─ PhaseEncoding ──────────────────────────────────────────
    │    D = W_re·x + i·W_im·x
    │    x ∈ ℝ^d_model → D ∈ ℂ^{B×h×T×d_k}
    │    （实部 + 虚部 = 相位空间编码）
    │
    ├─ AdjacencyBuilder ───────────────────────────────────────
    │    Ã = softmax(QKᵀ/√d_k)
    │    （图骨架：token 之间的初始连接强度）
    │    可选 adj_topk：稀疏邻接，每个 token 只连最强的 k 个邻居
    │
    ├─ TopologyPropagation (n_steps 步，惯性计算) ────────────
    │    for step in range(n_steps):
    │        active = nonzero(|D - D_prev| / |D_prev| >= θ)
    │        D[active] = (1-λ)D[active] + λ(Ã·D)[active]
    │    D' = D + α·D_0    （残差跳接，梯度直通）
    │
    ├─ 相位同步注意力矩阵 ──────────────────────────────────────
    │    phase_sim[i,j] = Re(D'_i · D'_j*) / √d_k
    │    phase_attn = softmax(phase_sim)
    │    可选 phase_topk：稀疏坍缩，只保留最强同步的 k 对
    │
    └─ 值聚合 + 输出投影 ──────────────────────────────────────
         V = W_V · x
         out = phase_attn · V    （跨 token 信息混合）
         return W_O · out
```

### 2.4 核心方程

**相位传播**（拓扑传播 = OTAL 动力学的矩阵形式）：
```
D(t+1) = (1-λ) · D(t) + λ · Ã · D(t)
```

**相位同步注意力**（Hermitian 内积 = Kuramoto 相位对齐度的矩阵形式）：
```
phase_attn[i,j] = softmax_j( Re(D'_i · conj(D'_j)) / √d_k )
```

物理含义：token i 与 token j 经过 n_steps 传播后的相位对齐程度，决定它们互相关注的强度。相位锁定 → 强烈互相关注。

### 2.5 可学习参数

| 参数 | 含义 | 初始值 |
|---|---|---|
| W_re, W_im | 相位编码权重 | Xavier |
| W_Q, W_K | 邻接矩阵构建 | Xavier |
| W_V, W_O | 值投影与输出 | Xavier |
| logit_λ | 传播耦合系数 | log(0.05/0.95) ≈ -2.94（warmup） |
| log_α | 残差跳接系数 | -3.0（近似 0） |
| logit_θ | 惯性阈值 | -4.0（近似 0.018，训练初期不冻结） |

λ 从 0.05 开始（近恒等变换），让梯度在训练初期直接流过；随训练进行 λ 增大，相位动力学逐渐显现。

### 2.6 与 Transformer 的差异

```
Transformer:
  step 1: score = softmax(QKᵀ / √d)       ← 一次性手工计算相似度
  step 2: out   = score · V                ← 直接加权求和

Oscillator:
  step 1: D = W_re·x + i·W_im·x           ← 编码进相位空间
  step 2: D' = propagate(D, Ã, n_steps)   ← n_steps 步相位传播（动力学）
  step 3: phase_attn = softmax(Re(D'·D'†)/√d_k)  ← 从动力学中涌现的注意力
  step 4: out = phase_attn · V             ← 加权求和（与 Transformer 相同结构）
```

Transformer 的注意力是手工设计的相似度度量。
Oscillator 的注意力是拓扑传播动力学的自然结果——不是计算出来的，而是涌现出来的。

### 2.7 实验结果（HLBD，300 steps，GPU）

| 模型 | 最终 loss | 备注 |
|---|---|---|
| Transformer | 0.606 | 基线 |
| Oscillator dense | 1.082 | 稠密相位传播 |
| Oscillator sparse k=12 | 1.057 | 稀疏相位同步，T/4 最强邻居 |

Oscillator 收敛速度约为 Transformer 的 1/2，符合预期——相位传播动力学比直接 QKᵀ 优化难度更高，λ warmup 机制使得训练初期相位动力学尚未完全生效。

---

## 三、系统关系图

```
                    量子大模型（QLM）
                         │
              ┌──────────┴──────────┐
              │                     │
        物理演化基底             可微分矩阵基底
        QCU / IQPU               Oscillator
              │                     │
    Lindblad + RK4            OTAL 矩阵等价形式
    dρ/dt = -i[H,ρ] + ΣD[c]ρ  phase_attn = softmax(Re(D'D'†)/√d)
              │                     │
         OTAL（桥梁）────────────────┘
         快速预筛选：89.8× 加速
         graph上的 Kuramoto 动力学
              │
         指令调度层（QCU ISA）
         qcu_lang / phase_map / compiler
              │
         应用层
         hash_search / collapse_scan / ...
```

OTAL 是连接两个基底的桥梁：
- 在 QCU 侧：作为物理仿真的预筛选器，不需要梯度，只需要快
- 在 Oscillator 侧：被重新诠释为可微分注意力动力学，必须可微

---

*文档版本：2026-03-29*
*基于 QCU v6、Oscillator v3（Hermitian attention + inertial computation）*
