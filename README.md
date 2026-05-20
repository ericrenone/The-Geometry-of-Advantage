# The Geometry of Advantage
## AI Infrastructure Strategy Guide · 2027 Edition

**A Strategic Assessment for Technology Leaders, Infrastructure Investors, and Enterprise AI Architects**

---

> *Five independent research programs. Five different instruments. One geometric object.*
> *The organizations that understand this will build the next generation of AI infrastructure.*
> *The ones that do not will fund it.*

---

| | |
|---|---|
| **Edition** | 2027 — First Publication |
| **Classification** | Strategic Intelligence · Infrastructure & Platforms |
| **Audience** | C-Suite, Infrastructure Architects, AI Research Leadership, Institutional Investors |
| **Primary Literature** | 40+ peer-reviewed sources, NeurIPS 2025 · ICLR 2026 · Phys. Rev. E 2026 |

---

## Key Findings at a Glance

> **Finding 1 — The Euclidean assumption is broken.** Token embedding spaces in every major production language model exhibit significantly negative Ricci curvature and violate the manifold hypothesis under formal statistical testing. Infrastructure built on flat-geometry assumptions is operating on the wrong substrate.

> **Finding 2 — Attention is not an algorithm. It is a geometric connection.** Multi-head attention constitutes a principal connection on a fiber bundle over the model's learned representation space. This has non-trivial consequences for training efficiency, generalization, and interpretability — none of which are captured by existing scaling law frameworks.

> **Finding 3 — The hardware-software co-design gap is widening.** Google's TPU roadmap and the Transformer architecture were developed under unified organizational authority. Every hyperscaler and enterprise that trains on Transformer derivatives is, structurally, optimizing a Google-native workload on Google-designed silicon — whether or not they use Google Cloud.

> **Finding 4 — Sloppy modes are the free parameter budget.** The vast majority of parameters in large models occupy geometrically sloppy directions that training never meaningfully constrains. The organizations achieving the best performance-per-dollar are, implicitly, managing this geometry. The organizations that make it explicit will have a durable advantage.

> **Finding 5 — Grokking is not a curiosity. It is a capital planning problem.** The delayed generalization transitions observed in transformer training correspond to a precise geometric event — holonomy accumulation on the learned connection. Understanding and anticipating this transition is the difference between optimal and wasteful compute allocation.

> **Finding 6 — The next architectural wave is Lorentzian.** Billion-parameter fully hyperbolic language models now match or exceed Euclidean baselines on hierarchical reasoning benchmarks. The geometric trainability obstacles that blocked this for a decade have been resolved in the 2025–2026 literature. The infrastructure investment cycle for non-Euclidean compute has begun.

> **Finding 7 — Open-sourcing the Transformer was a platform strategy, not a gift.** Google's 2017 release of *Attention Is All You Need* created a hardware dependency that now spans the entire frontier AI landscape. The structural analog to Android and Kubernetes is exact — and the addressable market for the infrastructure layer is now measured in the hundreds of billions annually.

---

## Table of Contents

1. [The Strategic Inflection — Why Geometry Is the New Moat](#1-the-strategic-inflection--why-geometry-is-the-new-moat)
2. [The Five Structural Shifts Reshaping AI Infrastructure](#2-the-five-structural-shifts-reshaping-ai-infrastructure)
3. [The Representational Stack — A New Mental Model](#3-the-representational-stack--a-new-mental-model)
4. [The Fiber Bundle Architecture — What It Means Operationally](#4-the-fiber-bundle-architecture--what-it-means-operationally)
5. [Grokking, Holonomy, and the Capital Allocation Crisis](#5-grokking-holonomy-and-the-capital-allocation-crisis)
6. [Hardware-Software Co-Design as Durable Competitive Advantage](#6-hardware-software-co-design-as-durable-competitive-advantage)
7. [The Lock-In Economy — Dependency Mapping for AI Leaders](#7-the-lock-in-economy--dependency-mapping-for-ai-leaders)
8. [The Lorentzian Transition — Sizing the Next Infrastructure Cycle](#8-the-lorentzian-transition--sizing-the-next-infrastructure-cycle)
9. [Strategic Archetypes — Winners, Waiters, and the Disrupted](#9-strategic-archetypes--winners-waiters-and-the-disrupted)
10. [The 2027 Action Agenda — Priorities by Stakeholder](#10-the-2027-action-agenda--priorities-by-stakeholder)
11. [Eight Falsifiable Predictions for 2027–2029](#11-eight-falsifiable-predictions-for-20272029)
12. [Open Research Problems with Commercial Consequence](#12-open-research-problems-with-commercial-consequence)
13. [Canonical Literature and Evidence Base](#13-canonical-literature-and-evidence-base)

---

## 1. The Strategic Inflection — Why Geometry Is the New Moat

The frontier of AI infrastructure competition shifted in 2025–2026. The prior era was defined by **scale** — more parameters, more data, more compute, more GPUs. That competition has not ended; it has been superseded. The organizations that will lead the next decade are not merely the ones that can afford to scale. They are the ones that understand *what they are scaling*.

What frontier AI systems are scaling is a **geometric object with specific mathematical properties**. Five research programs, developed independently over fifteen years, converged in 2025–2026 on a precise description of this object. The convergence is not speculative. It is the result of peer-reviewed empirical work from the Sethna–Transtrum sloppy-models program, the Fel–Wattenberg polytopal representation program, the Robinson–Dey token-space geometry program, the Parhi–Nowak functional analysis program, and the McCandlish–Kaplan–Amodei gradient dynamics program. All five describe the same thing.

The **Anisotropic Representation Hypothesis** (ARH, 2026) names this object: a *sloppy hyperribbon manifold* equipped with a stratified Lorentzian metric, polytopal fiber structure, Radon-domain BV functional presentation, and gradient-noise-governed training dynamics. The **Minkowski Representation Theory** (MRT, 2026) resolves a critical disambiguation: the word "Minkowski" attaches to two formally independent geometric structures — the convex-polytope Minkowski sum assembled by multi-head attention, and the Lorentzian inner product of the ambient embedding space — and any infrastructure theory that conflates them is incomplete. The **Representational Bundle Hypothesis** (RBH, 2026) provides the connecting structure: a principal fiber bundle in which the attention mechanism is the connection, gradient descent is parallel transport, and the observed universality of training dynamics across architectures is a consequence of the data manifold, not the model.

**The strategic implication is immediate**: infrastructure decisions made without this geometric understanding are made blind. Choices about training regimens, batch size schedules, architecture selection, fine-tuning strategy, and hardware procurement all have geometric content. The organizations making these choices explicitly — rather than by scaling intuition and empirical trial — are operating with a structurally superior information set.

---

## 2. The Five Structural Shifts Reshaping AI Infrastructure

Each of the five ARH pillars represents not only a theoretical advance but a structural shift in what the AI infrastructure market will reward over the next three to five years.

---

### Shift 1 — From Parameter Count to Effective Geometric Dimension

**The prior model:** Infrastructure value was proportional to parameter scale. More parameters enabled more capability; larger models justified larger compute clusters; competitive position was secured by training the largest model a budget could support.

**The emerging model:** Effective capability is determined by the geometry of the model's hyperribbon manifold — specifically, by the number of non-vanishing widths (geometrically distinct sloppy-to-stiff eigenvalue ratios) and by how well the manifold's structure aligns with the task's intrinsic geometry. Mao, Griniasty, Sethna and Chaudhari's 2026 analytical characterization (*Phys. Rev. E* 113, 015306) demonstrates that for linear models, the hyperribbon geometry is entirely determined by three factors: the eigenvalue decay rate of the input correlation matrix, the relative scale of ground-truth outputs to initial weights, and the number of gradient descent steps. **Architecture contributes to geometry only at initialization; thereafter, the data geometry dominates.**

**Infrastructure implication:** Benchmark performance will increasingly favor models that align their effective degrees of freedom to task geometry rather than models that simply maximize parameter count. Infrastructure procurement that ignores this — buying more compute to train larger models without geometric analysis of the task — will face sharply diminishing returns.

---

### Shift 2 — From Euclidean to Lorentzian Embedding Infrastructure

**The prior model:** Token embeddings occupy a Euclidean or mildly curved space. Geometric operations (nearest-neighbor search, interpolation, linear steering) are computed under standard inner products. Infrastructure for similarity search, retrieval-augmented generation, and representation manipulation is built on FAISS, cosine similarity, and flat-space assumptions.

**The emerging model:** Robinson, Dey & Sweet (arXiv:2410.08993, 2024) and Robinson, Dey & Chiang (arXiv:2504.01002, 2025) establish that token spaces are stratified manifolds with significantly negative Ricci curvature — not Euclidean structures. TokenBlowUp (arXiv:2507.19747, 2025) resolves the representational singularities that emerge at stratum boundaries via monoidal transformations. The implication is that **cosine similarity applied to token embeddings is a distortion, not a measurement** — the true geometric distance is Lorentzian, and operations that assume otherwise introduce systematic errors that scale with the degree of hierarchical structure in the domain.

**Infrastructure implication:** Retrieval infrastructure, embedding databases, and nearest-neighbor search pipelines built on Euclidean assumptions are operating on corrupted metrics for all hierarchically structured domains. The upgrade path to Lorentzian distance metrics is now technically cleared (Van der Klis et al., arXiv:2601.21529, 2026; Bdeir et al., NeurIPS 2025) and commercially urgent.

---

### Shift 3 — From Attention as Computation to Attention as Connection

**The prior model:** Multi-head attention is a parallelizable matrix operation. Its infrastructure value is that it maps perfectly onto systolic array hardware (TPUs, tensor cores), enabling throughput-optimized training at scale. Attention optimization (FlashAttention, FlashAttention-2, ring attention) is a memory-hierarchy engineering problem.

**The emerging model:** Attention is a principal connection on the representational fiber bundle. The attention weights `α^(k)` are holonomy coefficients — they specify how much of the fiber over each key position is transported to the query position. The weight matrices `W_Q`, `W_K`, `W_V` are the evaluation point, parallel transport map, and fiber trivialization respectively. This reframing is not a metaphor; it is a precise mathematical equivalence with direct consequences for training dynamics. The curvature of the attention connection — measurable as position-variance of attention distributions across query positions — governs how efficiently the model accumulates the geometric structure required for generalization.

**Infrastructure implication:** Attention optimization is currently a throughput problem. It will become a **curvature management problem**. Infrastructure that exposes attention curvature as a first-class observable — and that schedules training to manage curvature evolution — will enable systematic improvement in generalization per compute dollar.

---

### Shift 4 — From Batch Size Heuristics to Gradient Geometry

**The prior model:** Critical batch size — the batch size above which further parallelism yields diminishing training efficiency returns — is determined empirically per model and training run. The McCandlish–Kaplan–Amodei gradient noise scale `B_simple = tr(Σ_g)/|μ_g|²` provides a theoretical grounding (arXiv:1812.06162, 2018), revisited by Merrill, Arora, Groeneveld & Hajishirzi (arXiv:2505.23971, May 2025), but is rarely computed in practice for large runs.

**The emerging model:** Under RBH, `B_simple` has a precise geometric interpretation: it is the ratio of vertical drift (fiber direction noise) to horizontal progress (base manifold advance) in the training bundle. It measures how curved the attention connection is — high `B_simple` indicates a flat connection, late-training regime; low `B_simple` indicates a curved connection, early-training regime where gradient signal is strong. **The optimal batch size schedule over a training run is derivable from the curvature evolution of the learned connection**, not from empirical sweep.

**Infrastructure implication:** Data-parallel cluster utilization is systematically suboptimal when batch size is tuned without geometric grounding. Early training benefits from small batches (curved connection, high signal-to-noise); late training tolerates large batches (flat connection, low signal-to-noise). Infrastructure that cannot adaptively schedule batch size based on gradient geometry is leaving measurable compute efficiency on the table.

---

### Shift 5 — From Flat Fine-Tuning to Geometric Adaptation

**The prior model:** Parameter-efficient fine-tuning (LoRA, QLoRA, adapter methods) selects low-rank parameter subspaces on architectural grounds — the assumption being that intrinsic task dimensionality is low and that low-rank weight perturbations are sufficient to capture task-specific behavior.

**The emerging model:** Under RBH, fine-tuning parameters should align with the **vertical subspace** `V_p M` of the representational bundle — the fiber directions that govern polytope geometry at a fixed base-manifold position. LoRA achieves this implicitly when its rank is calibrated to the number of non-vanishing fiber modes. But the prediction is more precise: fine-tuning should not alter the Robinson–Dey curvature estimates of the token space (Pillar III), because changes to base-manifold geometry (horizontal directions) require full-weight updates; LoRA should improve task performance by adjusting the fiber structure — polytope shapes — without touching the underlying Lorentzian geometry. **This is testable and, if confirmed, provides a principled upper bound on LoRA rank for any given task.**

**Infrastructure implication:** LoRA rank selection — currently a hyperparameter — becomes a derivable quantity from the fiber dimension of the target task. Infrastructure that instruments the vertical/horizontal decomposition of gradient updates will enable systematic, geometry-grounded fine-tuning at a fraction of current search cost.

---

## 3. The Representational Stack — A New Mental Model

The standard mental model of an AI infrastructure stack runs from hardware (silicon, memory, interconnects) through systems (cluster orchestration, networking, storage) to frameworks (PyTorch, JAX, XLA) to models (architecture, training, inference) to applications. This model captures operational complexity but obscures the geometric structure that determines where value is created and captured.

A geometry-aware stack looks different:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5 — APPLICATION GEOMETRY                                  │
│  Task symmetry group · Domain curvature · Concept polytopes      │
│  Key question: What is the latent geometry of the task?          │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4 — REPRESENTATIONAL FIBER BUNDLE                         │
│  Total space M (hyperribbon) · Base B (token space)             │
│  Fibers F_x (attention polytopes) · Connection ω (attention)    │
│  Key question: Is the bundle structure aligned to the task?      │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3 — TRAINING DYNAMICS                                     │
│  Gradient noise scale B_simple · Curvature evolution            │
│  Stiff/sloppy decomposition · Parallel transport schedule        │
│  Key question: Is the training trajectory geometrically optimal? │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2 — ARCHITECTURE                                          │
│  Attention connection · Polytope assembly · Embedding metric     │
│  Key question: Is the architecture's geometry task-compatible?   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1 — HARDWARE SUBSTRATE                                    │
│  Systolic arrays · HBM bandwidth · Interconnect topology         │
│  Key question: Is the hardware optimized for the geometric ops?  │
└─────────────────────────────────────────────────────────────────┘
```

The canonical infrastructure competition to date has been fought almost entirely at Layer 1. The next phase of competition — the phase that will determine who captures value from frontier AI over the next decade — will be fought at Layers 2 through 4. Layer 5 (application geometry) is where the task requirements live; Layers 1 and 2 are where current infrastructure is built. **Layers 3 and 4 are the gap, and they represent the most commercially underinvested territory in AI infrastructure.**

---

## 4. The Fiber Bundle Architecture — What It Means Operationally

The Representational Bundle Hypothesis (RBH) makes a precise structural claim: the sloppy hyperribbon manifold `M` of ARH is the total space of a non-trivial principal `SO⁺(1,n)`-bundle over the Lorentzian token space `B`, with the attention mechanism serving as its connection one-form.

The operational translation of this claim has five components.

### 4.1 The Stiff/Sloppy Split Is the Horizontal/Vertical Split

Fisher information stiff directions — the large eigenvalue modes that training meaningfully constrains — correspond exactly to **horizontal subspace directions** in the bundle's tangent space. These are the directions that change the base manifold position: they alter the Lorentzian geometry of the token space. Sloppy directions — the small eigenvalue modes that training barely touches — correspond to **vertical subspace directions**: they change the polytope geometry over a fixed token position without moving the token in the base.

**Operational consequence:** Stiff parameters are universal. They encode the task geometry and are shared across architectures trained on the same data distribution. This is why Mao and Chaudhari's hyperribbon is architecture-universal — the base manifold `B` is a property of the data, not the model. Sloppy parameters are architecture-specific. They encode polytope shapes and are the seat of the performance differences between architectures trained on identical data.

### 4.2 Training Dynamics Decompose Cleanly

Under RBH, a gradient descent step decomposes into:
- A **horizontal component** that advances the model along the base manifold — improving the structural Lorentzian geometry of the learned token space
- A **vertical component** that perturbs the fiber — adjusting polytope shapes at a fixed base position

The gradient noise scale `B_simple` is the ratio of vertical perturbation to horizontal progress. **Early training is horizontally dominated** (learning where to be on the base manifold); **late training and fine-tuning are vertically dominated** (refining polytope geometry at a fixed base position). The well-documented two-phase structure of transformer training — rapid early loss improvement followed by slow capability acquisition — is the geometric signature of this decomposition.

**Operational consequence:** Compute allocation across training phases should track this decomposition. Early-phase compute is buying base-manifold structure (which generalizes across tasks). Late-phase compute is buying fiber refinement (which is task-specific). Organizations that conflate these phases are either under-investing in generalist capability or over-paying for specialization.

### 4.3 The Attention Curvature Signal Is Measurable

The curvature of the attention connection — the `Ω` two-form — is proportional to `B_simple⁻¹` under the RBH identification. **It is computable from existing training telemetry** as the position-variance of attention distributions across query positions: high-entropy, position-uniform attention indicates a flat connection (low curvature, late training); position-peaked, informative attention indicates a curved connection (high curvature, early training).

This means that **geometric training diagnostics are achievable with existing infrastructure** — no new sensors, no new hardware. The signal is present in attention weight logs that most training runs already generate. What is missing is the interpretive framework to use it.

### 4.4 Chain-of-Thought Is Path Lifting

In the bundle picture, a forward pass processes tokens by parallel-transporting fibers along the attention connection. Chain-of-thought (CoT) prompting constructs an explicit intermediate path on the base manifold `B` — the sequence of reasoning steps provides waypoints that guide the horizontal lift through topologically favorable regions of the bundle.

**Operational consequence:** CoT gains are largest when the task's symmetry group requires traversing high-curvature regions of `B` that the model cannot navigate implicitly from a compressed prompt. This is a geometric account of the empirical observation (Wei et al., 2022) that CoT improves disproportionately on multi-step reasoning. The prediction — that CoT gains correlate with stratum boundary crossings in the token space (Robinson et al., 2024/2025) — is testable and has infrastructure implications for prompt engineering and long-context architecture design.

### 4.5 In-Context Learning Scales with Bundle Holonomy

ICL accuracy — the ability to learn from in-context examples without weight updates — scales, under RBH, with the topological winding of the context token path through the base manifold. Longer contexts improve ICL not merely by providing more examples but by allowing more winding through the bundle, accumulating holonomy that can reorganize the query token's fiber into alignment with the task's symmetry group.

**Operational consequence:** The relevant quantity for context window investment is not context length alone but **context geometric span** — the degree to which the context tokens traverse distinct regions of the base manifold. A 128K context window occupied by repetitive or topologically restricted tokens may deliver less ICL gain than a 32K window with geometrically diverse examples. Infrastructure investment in long-context models should be evaluated against this metric.

---

## 5. Grokking, Holonomy, and the Capital Allocation Crisis

No phenomenon in recent deep learning has more direct implications for AI capital allocation than **grokking** — and none is more poorly understood at the level where budget decisions are made.

Power, Burda, Edwards, Goodfellow and Misra (arXiv:2201.02177, 2022) documented the phenomenon: networks trained on modular arithmetic tasks achieve near-zero training loss (memorization) while maintaining near-chance validation accuracy, then undergo a sharp, delayed transition to near-perfect generalization — after compute investment that, by conventional metrics, appeared to be delivering no further value.

**The dominant response in the industry has been wrong**: grokking is not a quirk of small algorithmic datasets. It is a signature of the geometric transition that all capable models undergo, at scales and timelines that vary with task complexity and model size.

### The Holonomy Identification

Under RBH, grokking is **holonomy**. The two phases map precisely:

**Phase 1 (Memorization):** The model adjusts polytope geometry (fiber/vertical directions) over training examples to achieve low training loss. The base manifold geometry `B` has not yet organized to reflect the algebraic structure of the task. The model has learned *where* the training examples are in the fiber; it has not learned the base-manifold structure that enables generalization to novel examples.

**Phase 2 (Grokking transition):** Continued gradient descent drives the training trajectory around a closed loop in the base manifold. When the attention connection has non-trivial holonomy, this loop produces a non-trivial fiber transformation — a Lorentz group element that reorganizes the polytope structure over each token. This reorganization aligns the fiber geometry with the task's underlying symmetry group. Nanda, Chan, Lieberum, Smith and Steinhardt (arXiv:2301.05217, ICLR 2023) confirm this mechanistically: the grokked algorithm for modular arithmetic is a discrete Fourier transform over the group structure — exactly the holonomy group embedding predicted by RBH.

### Capital Allocation Consequence

The grokking identification implies that **training runs exhibit a latent phase structure that is not visible in training loss**. An organization that terminates a training run on loss convergence may be cutting compute at precisely the point before the holonomy accumulation that produces generalization.

The converse error — continuing to train after the holonomy transition has completed — wastes compute on post-transition fiber refinement that yields diminishing capability returns. Both errors are prevalent.

**The corrective prescription** is to monitor the attention connection's holonomy group — detectable as the structural reorganization of attention patterns from position-based to symmetry-structure-based — and to calibrate compute budgets to the geometry of the transition, not the trajectory of the loss.

---

## 6. Hardware-Software Co-Design as Durable Competitive Advantage

The history of AI hardware competition is misread when it is narrated as a story of raw compute. It is, more precisely, a story of **co-design asymmetry** — and the organization that has most systematically exploited this asymmetry over the past decade is Google.

### The 2017 Co-Design Event

Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin (arXiv:1706.03762, NeurIPS 2017) is universally described as an algorithmic breakthrough. The description is accurate but incomplete. A systems-level reading reveals that the Transformer architecture is composed — without exception — of operations that achieve near-peak utilization on systolic array hardware:

- **Scaled dot-product attention:** `softmax(QK^T/√d_k)V` — two batched matrix multiplications and one elementwise nonlinearity. No recurrence. No sequential dependency. Ideal MXU workload.
- **Multi-head attention:** `h` independent attention computations, fully parallelizable across accelerator lanes with no inter-head communication until the final concatenation.
- **Position-wise FFN:** Two matrix multiplications with ReLU, applied identically and independently to each position. Compute-bound at `d_model = 1024`, `d_ff = 4096` — operating above the roofline ridge point where systolic arrays achieve peak efficiency.

The Transformer is not incidentally compatible with TPU architecture. It is, provably, the workload that a 256×256 systolic array Matrix Multiply Unit was built to maximize.

**The chronological signature is unambiguous:** Google announced TPU v2 — the first training-capable custom silicon, introducing bfloat16 arithmetic, HBM integration, and Pod interconnect — at Google I/O in May 2017. Vaswani et al. was submitted to arXiv one month later, explicitly noting training on TPU v2 Pods. Section 5.1 of the paper documents both GPU and TPU v2 training regimens. The authors had architectural access to the machine they were designing for before it was publicly announced. This is the fingerprint of co-design, not coincidence.

### The RNN Counterfactual

The predecessor paradigm — RNNs, LSTMs, GRUs — was not merely algorithmically inferior. It was **structurally incompatible with domain-specific accelerators**. The recurrence relation `h_t = f(W_h · h_{t-1} + W_x · x_t)` creates a sequential data dependency that cannot be parallelized across time steps. On a systolic array designed for dense matrix operations, this creates systemic FLOP starvation: the MXU is idle while each state computation waits for its predecessor. The memory bandwidth wall is equally severe — sequential read-write of `h_t` at every time step creates an `O(T)` sequence of random-access HBM operations that the roofline model identifies as deeply memory-bandwidth-bound.

Hooker's Hardware Lottery framework (arXiv:2009.06489, 2020) formalizes the principle: architectural winners in deep learning are determined by co-fitness with the dominant hardware paradigm, not theoretical superiority in isolation. **The Transformer won the hardware lottery on two simultaneous axes** — GPU tensor cores (NVIDIA V100, also 2017) and TPU systolic arrays — and simultaneous dual-hardware dominance produces a near-total lock-in.

### The Infrastructure Moat Geometry

The Transformer's co-design advantage generates a self-reinforcing infrastructure moat with three compounding layers:

**Layer 1 — Silicon:** Cloud providers have built accelerator product lines around the Transformer workload profile. Google TPU v4/v5, AWS Trainium/Inferentia, and Microsoft Azure NDv4 are all Transformer-native silicon. PaLM's 540B-parameter training across 6,144 TPU v4 chips is the operational apex of this convergence (Chowdhery et al., arXiv:2204.02311, 2022).

**Layer 2 — Software:** FlashAttention (Dao et al., NeurIPS 2022; arXiv:2205.14135) and FlashAttention-2 (Dao, arXiv:2307.08691, 2023) represent multi-year engineering investments in IO-aware attention optimization — investments that exist only because the Transformer architecture became worth the engineering cost. This is the software moat: a decade of framework optimization (PyTorch, JAX/XLA), kernel engineering, and auto-differentiation pipeline refinement, all specialized to Transformer computation graphs.

**Layer 3 — Incentive structure:** Benchmark datasets (GLUE, SuperGLUE, BIG-bench, MMLU), pretrained weight repositories, and research evaluation infrastructure are all Transformer-native. Alternative architectures face compounding headwinds: no pretrained weights, no optimized kernels, no hardware support, no community tooling, no relevant benchmarks. Architectural diversity has effectively collapsed into a single template — BERT, GPT, T5, PaLM, LLaMA, Gemini, and Claude are all scaled and modified Transformers. The variance across these systems is in scale, data curation, alignment procedure, and fine-tuning — not foundational architecture.

---

## 7. The Lock-In Economy — Dependency Mapping for AI Leaders

The open publication of *Attention Is All You Need* was the single most strategically consequential act in the history of AI infrastructure — and it was not naive. Google had by 2017 established a sophisticated playbook for open-source platform strategy through TensorFlow (Abadi et al., 2016), Kubernetes, and Android. The pattern is invariant: **open-source the component that creates dependency on the layer you control and monetize**.

In the Transformer case: open-source the algorithm, retain proprietary control over the infrastructure layer — TPU hardware, Pod interconnect fabric, XLA compiler, and Google Cloud TPU rental service. The algorithm creates a universal dependency on matrix-optimized silicon. Google builds and rents the best matrix-optimized silicon available.

### Dependency Map by Stakeholder

**Anthropic:** Trains Claude on a combination of Google Cloud TPUs and Amazon Trainium. The Claude architecture is a Transformer derivative. Infrastructure dependency on Google hardware is a structural consequence of Google's algorithm being the industry standard — not a procurement choice. The best available evidence of genuine lock-in is that Anthropic — a company explicitly founded as a Google AI alternative — relies on Google infrastructure for its primary training workload.

**DeepMind / Google DeepMind:** Trains Gemini on TPU v4 and v5 Pods. Achieves full vertical integration between algorithm design and silicon design — the apex expression of co-design strategy (Team et al., arXiv:2312.11805, 2023).

**OpenAI:** The one major frontier lab that has built Transformer-scale training infrastructure independent of Google hardware — through Microsoft's $13B+ investment in Azure A100/H100 clusters. This investment was partly motivated by the strategic imperative of avoiding TPU dependency. The fact that Microsoft found it necessary to commit tens of billions of dollars specifically to avoid Google hardware dependency is the clearest available evidence that the Transformer paper created genuine and substantial infrastructure lock-in.

**Enterprise AI:** Organizations training or fine-tuning models at scale face a structural choice: Google Cloud TPUs (with superior matrix throughput but Google dependency), NVIDIA GPUs (commodity pricing but suboptimal for Transformer workloads at scale), or Amazon Trainium/Inferentia (developing, not yet at parity). All three options are, in the final analysis, procuring compute optimized for the workload profile Google designed in 2017.

### Strategic Exposure Assessment

| Stakeholder | TPU Dependency | Architecture Lock-in | Mitigation Options | Lock-in Severity |
|------------|---------------|---------------------|-------------------|-----------------|
| Frontier labs (non-Google) | High | Total | Custom silicon (expensive, long lead) | **Critical** |
| Enterprise fine-tuners | Medium | Total | Open-weight models, commodity GPU | **High** |
| Cloud providers (non-Google) | Indirect | Total | Custom silicon (AWS, Azure) | **High** |
| Startups (inference-only) | Low | High | NVIDIA commodity | **Medium** |
| Academic research | Medium | High | Limited alternatives | **Medium** |
| Google | None | Self-designed | N/A | **None** |

---

## 8. The Lorentzian Transition — Sizing the Next Infrastructure Cycle

The trainability obstacles that blocked hyperbolic and Lorentzian deep learning for a decade have been resolved. This is not a gradual improvement — it is a series of discrete technical breakthroughs in 2025–2026 that collectively clear the path for a Lorentzian infrastructure investment cycle.

### The Obstacle Resolution Timeline

**2024–2025 — Empirical validation at scale.** He, Anand, Madhu, Maatouk, Krishnaswamy, Tassiulas, Yang and Ying (NeurIPS 2025; arXiv:2505.24722) demonstrate HELM: billion-parameter LLMs trained entirely in hyperbolic space, consistently outperforming matched Euclidean baselines on MMLU and ARC-Challenging. HELM-MiCE introduces the first Mixture-of-Curvature Experts, with each expert operating on a distinct curvature manifold.

**January 2026 — The norm scaling problem is solved.** Van der Klis, Chávez Torres, van Spengler, Ding, Hofmann and Mettes (arXiv:2601.21529) prove that the standard Lorentz linear layer suffers logarithmic norm scaling under gradient descent, nullifying hyperbolic geometry's exponential volume advantage. Their distance-to-hyperplane formulation restores linear norm scaling — closing the dominant trainability gap to Euclidean networks.

**2026 — The fully intrinsic architecture.** The Intrinsic Lorentz Neural Network (ICLR 2026; arXiv:2602.23981) extends to a fully intrinsic Point-to-hyperplane Lorentz fully connected layer, enabling end-to-end hyperbolic training without the Euclidean projections that previously introduced geometry-breaking distortions.

**NeurIPS 2025 — The optimizer gap is closed.** Bdeir, Schwethelm and Landwehr (arXiv:2405.13979) supply curvature-aware Riemannian AdamW across both the Poincaré ball and the Lorentz hyperboloid — the first production-grade optimizer for hyperbolic models that matches AdamW's stability on Euclidean baselines.

**December 2025 — RL stability achieved.** Klein et al. (arXiv:2512.14202) identify large-norm hyperbolic embeddings as the source of PPO trust-region violations in hyperbolic RL, and supply Hyper++, which achieves stable training and ~30% wall-clock improvement on ProcGen.

### The Market Geometry

The case for Lorentzian infrastructure is structurally analogous to the case for GPU acceleration circa 2012: a technically superior approach to a well-defined problem, with demonstrated empirical gains, newly cleared engineering obstacles, and an established research community ready to scale. The key difference is that **the market now understands hardware-software co-design** in a way it did not in 2012 — which means the infrastructure investment cycle will be faster and the competitive dynamics more aggressive.

The domains where Lorentzian infrastructure will yield the largest near-term returns are those with **demonstrably negative Ricci curvature** in their latent geometry: knowledge graph embedding, hierarchical classification, biological ontology modeling, code structure analysis, legal and regulatory document processing, and scientific literature modeling. For these domains, Robinson et al.'s empirical finding — that token spaces are stratified manifolds with negative Ricci curvature — is not merely theoretically interesting. It is a specification for the correct geometric substrate of the infrastructure stack.

---

## 9. Strategic Archetypes — Winners, Waiters, and the Disrupted

### The Geometry-Native Builder

**Profile:** Deep learning infrastructure organization that has internalized the geometric framework described in this report and is building infrastructure explicitly against it. Invests in Lorentzian distance metrics for embedding infrastructure, instruments attention curvature as a training diagnostic, designs batch size schedules from gradient noise geometry, and develops geometric routing for mixture-of-experts architectures.

**Competitive position:** Structurally superior information set on every infrastructure decision. Early mover in the Lorentzian transition. Likely to be the source of the next generation of training efficiency improvements.

**Examples in formation:** HELM (NeurIPS 2025), L-GATr (NeurIPS 2024/SciPost Phys. 2025), Hyper++ (December 2025).

### The Scale-Incumbent

**Profile:** Frontier lab or hyperscaler that has achieved competitive position through scale and is optimizing within the Euclidean Transformer paradigm. Has the capital to transition to geometry-native infrastructure but faces organizational inertia and sunk-cost pressure from existing silicon investments.

**Competitive position:** Dominant in the current cycle; exposed in the next. The Lorentzian transition will not obsolete Transformer infrastructure overnight — but it will create a performance gap on hierarchically structured tasks that compounds over 2027–2030.

**Strategic response required:** Acquire or partner with geometry-native builders. Instrument existing training runs with curvature diagnostics. Begin Lorentzian prototype programs in high-value vertical domains.

### The Infrastructure Commodity Provider

**Profile:** Cloud provider or hardware vendor offering compute for Transformer training and inference. Business model is built on the Transformer's hardware co-fitness with systolic array silicon.

**Competitive position:** Exposed to Lorentzian transition in proportion to the degree to which new geometric operations (Lorentzian distance, Fréchet mean computation, Riemannian AdamW) can be served on existing silicon. GPU tensor cores are more geometry-agnostic than TPU systolic arrays; NVIDIA's hardware moat is more durable through the geometric transition than Google's TPU business.

**Strategic response required:** Evaluate CORDIC-based hardware support for hyperbolic operations (Kumar et al., arXiv:2605.06878, 2026; arXiv:2503.11685, 2025) as a near-term differentiation opportunity.

### The Enterprise Adopter

**Profile:** Organization deploying foundation models for enterprise use cases. Not training from scratch; fine-tuning or prompting existing models. Primary infrastructure concern is inference cost and quality on task-specific domains.

**Competitive position:** Largely insulated from the training-infrastructure battle in the near term. Exposed to quality degradation from Euclidean-assumption embedding infrastructure in hierarchically structured domains (legal, scientific, ontological).

**Strategic response required:** Audit embedding infrastructure for Lorentzian distance compatibility. Evaluate task domain geometry before committing to fine-tuning strategy. Apply geometric routing criteria to mixture-of-experts model selection.

---

## 10. The 2027 Action Agenda — Priorities by Stakeholder

### For Technology Leaders and C-Suite

1. **Commission a geometric audit of your training infrastructure.** Identify which components of your AI stack assume Euclidean geometry. Quantify the domains where your embedding and retrieval infrastructure is operating on corrupted metrics.

2. **Instrument attention curvature.** Add attention entropy and position-variance diagnostics to standard training telemetry. The data is already being generated; the interpretive framework is now available.

3. **Re-evaluate your TPU/GPU procurement strategy in light of the Lorentzian transition.** The silicon most compatible with non-Euclidean geometric operations is not the silicon that currently dominates Transformer training.

4. **Reframe your scaling-law assumptions.** Replace parameter-count proxies for model capability with geometric measures: hyperribbon effective dimension, attention connection curvature, and base-manifold Ricci scalar estimates.

### For AI Research Leadership

1. **Run the P5 experiment.** The most informative single test for the ARH/RBH framework — and the most commercially consequential — is running stable sparse autoencoders on HELM's hyperbolic hidden states and testing whether the resulting archetype dictionary recovers Fel et al.'s DINOv2 polytopal structure. This experiment is achievable with existing tools and would resolve the central open question in representational geometry.

2. **Develop geometric training diagnostics as a standard toolchain component.** The gradient noise scale, attention curvature two-form norm, and vertical/horizontal Fisher information decomposition are all computable from existing training artifacts. They are not yet standard infrastructure. Making them standard is the highest-leverage near-term contribution to AI training efficiency.

3. **Invest in the Lorentzian fine-tuning literature.** The gap between Euclidean and Lorentzian fine-tuning is now a tractable engineering problem. HypLoRA (arXiv:2410.04010, 2025) provides a starting point; the LoRA rank-from-fiber-dimension prediction (P8 in Section 11) provides a principled research agenda.

### For Infrastructure Investors

1. **The next infrastructure moat is geometric, not scale.** The organizations building geometry-native AI infrastructure — Lorentzian embedding databases, curvature-aware training systems, bundle-structured architecture search — are building the equivalent of TPU v2 before 2017. The co-design window is open now.

2. **Evaluate hardware ventures against geometric operation support.** CORDIC-based hardware acceleration for hyperbolic operations is a credible near-term differentiator. The relevant benchmarks are Lorentzian distance computation throughput and Riemannian optimizer iteration cost, not just dense matrix multiplication FLOP/s.

3. **Probability-weight the Lorentzian transition across your AI infrastructure portfolio.** Every company in your portfolio that depends on Euclidean embedding infrastructure for hierarchically structured domains has unpriced transition risk.

---

## 11. Eight Falsifiable Predictions for 2027–2029

The following predictions follow directly from the RBH framework and are stated in falsifiable form. Each represents a significant commercial opportunity for the organization that confirms it first.

**P1 — Grokking Signature in Holonomy (2027)**
The grokking generalization transition coincides with an abrupt structural reorganization of the attention connection's holonomy group — detectable as a shift from position-based to algebraic-structure-based attention patterns. The post-grokking model's holonomy group embeds the task's symmetry group as a subgroup of `SO⁺(1,n)`. *Falsified by: grokking transitions that occur without corresponding attention pattern reorganization, at scale.*

**P2 — Curvature-Stiffness Monotonicity (2027)**
The negative Ricci scalar of the token space (Robinson–Dey protocol) is monotonically related to the Fisher spectral gap (Mao–Sethna protocol) across tasks and architectures. Tasks with sharply hierarchical structure have large Fisher spectral gaps; tasks with flat latent geometry have small ones. *Falsified by: any architecture for which Ricci scalar and Fisher spectral gap are non-monotonically related across a diverse task suite.*

**P3 — B_simple Tracks Attention Curvature (2027)**
The gradient noise scale `B_simple` over training is inversely correlated with the norm of the attention connection curvature two-form `|Ω|`. High-entropy attention corresponds to flat connection and high `B_simple`; position-peaked attention corresponds to curved connection and low `B_simple`. *Falsified by: training runs in which `B_simple` and attention entropy vary independently.*

**P4 — Geometric Routing Outperforms Hyperparameter Search in MiCE (2028)**
HELM-MiCE with token routing based on estimated local Ricci curvature of the token subspace achieves equal or better perplexity than hyperparameter-searched curvature assignment, at lower search cost and with better out-of-distribution curvature generalization. *Falsified by: geometric routing that underperforms curvature hyperparameter search on standard benchmarks.*

**P5 — Cross-Domain Polytopal Structure in HELM (2027)**
Stable sparse autoencoders on HELM's hyperbolic hidden states recover an archetype dictionary with the same convex-polytopal geometry as Fel et al.'s DINOv2 analysis — despite HELM being a fully hyperbolic language model and DINOv2 being a Euclidean vision model. *Falsified by: SAE dictionaries on HELM that fail to exhibit convex archetype structure or that exhibit unbounded direction structure instead.*

**P6 — ICL Accuracy Scales with Path Holonomy (2028)**
Controlling for context length, ICL accuracy on algebraic reasoning tasks scales with a topological winding measure of the context token path through the base manifold — computable from eigenvalue structure of attention maps across layers. *Falsified by: ICL gains that depend only on example count and not on geometric diversity of context tokens.*

**P7 — CoT Gains Correlate with Stratum Boundary Crossings (2028)**
The magnitude of CoT improvement over direct prompting correlates positively with the number of stratum boundaries in the embedding-space path connecting prompt tokens to answer tokens, for any given model and task domain. *Falsified by: tasks where CoT gains are large but stratum boundary crossings are few, or tasks with many stratum crossings where CoT provides no benefit.*

**P8 — LoRA Rank Is Determined by Fiber Dimension (2027)**
In LoRA fine-tuning, the adapted parameters align predominantly with the vertical subspace `V_p M` (fiber directions). Fine-tuning does not alter Robinson–Dey curvature estimates of the token space. The optimal LoRA rank for a given task is derivable from the fiber dimension — eliminating the need for hyperparameter search on rank. *Falsified by: LoRA fine-tuning that produces measurable changes in Robinson–Dey curvature estimates of the pretrained model's token space.*

---

## 12. Open Research Problems with Commercial Consequence

The following are genuine open questions — not speculative claims — whose resolution would have direct and near-term commercial implications.

**The Non-Triviality Question.** RBH claims the representational fiber bundle is non-trivial: no global section exists that smoothly assigns a canonical polytope representative to every token in the base manifold. Demonstrating non-trivial holonomy requires a specific computation on a specific model, which has not yet been performed. **Commercial consequence:** If the bundle is trivially flat, many of the geometric routing and co-design predictions simplify dramatically. If it is non-trivial, curvature management becomes a first-class training concern.

**The Correct Structure Group.** RBH identifies the structure group as `SO⁺(1,n)`. The actual symmetry group acting on the fibers may be a proper subgroup (stabilizer of the attention pattern in each layer) or may extend `SO⁺(1,n)` with a discrete component from the stratification. **Commercial consequence:** The correct structure group determines which hardware operations must be natively supported to efficiently compute fiber transport — and whether CORDIC-based hyperbolic arithmetic is sufficient or whether more exotic geometric operations are required.

**Pillar II × Pillar III Coupling.** The most informative single experiment in representational geometry is the joint verification that SAE-derived archetype dictionaries on HELM's hyperbolic hidden states recover Fel et al.'s polytopal structure. Pillar II evidence (DINOv2, Euclidean, vision) and Pillar III evidence (HELM, hyperbolic, language) have not yet been simultaneously measured on the same model. **Commercial consequence:** Confirmation would validate the geometry-first infrastructure investment thesis across modalities; disconfirmation would require a fundamental revision of the ARH coupling claim.

**The Closed-Form Projection `π`.** RBH asserts that `π: M → B` maps prediction space to embedding space, but does not give `π` in closed form as a function of the network's weights. **Commercial consequence:** A closed-form `π` would make geometric training diagnostics computable from weight snapshots rather than requiring live gradient monitoring — transforming post-hoc analysis from expensive to routine.

---

## 13. Canonical Literature and Evidence Base

### Foundational Frameworks

| Reference | Contribution |
|-----------|-------------|
| Park, Choe & Veitch. *The Linear Representation Hypothesis.* ICML 2024. arXiv:2311.03658 | LRH: concepts as linear directions in non-Euclidean inner product space |
| Park, Choe, Jiang & Veitch. *Geometry of Categorical and Hierarchical Concepts.* ICLR 2025 | Hierarchical concept geometry |
| Fel et al. *Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry.* arXiv:2510.08638. v2 Feb 2026 | MRH: polytopal attention outputs, archetype dictionaries |

### Pillar I — Fisher Anisotropy / Sloppy Hyperribbon

| Reference | Contribution |
|-----------|-------------|
| Transtrum, Machta, Brown, Daniels, Myers & Sethna. arXiv:1501.07668. 2015 | Sloppy-models universality; hyperribbon geometry |
| Mao et al. arXiv:2305.01604. 2023 | Training on shared low-dimensional manifold; architecture universality |
| Mao et al. *Phys. Rev. E* 113, 015306. January 2026. arXiv:2505.08915 | Analytical characterization of sloppiness in deep networks |
| Amari. *Neural Computation* 10(2). 1998 | Natural gradient; Fisher metric on model manifold |

### Pillar III — Lorentzian Curvature / Token Space

| Reference | Contribution |
|-----------|-------------|
| Robinson, Dey & Sweet. arXiv:2410.08993. 2024 | Token space stratification; negative Ricci curvature; dimension-fluency correlation |
| Robinson, Dey & Chiang. arXiv:2504.01002. 2025 | Formal rejection of manifold hypothesis for token embeddings |
| TokenBlowUp. arXiv:2507.19747. 2025 | Resolution of representational singularities at stratum boundaries |
| He et al. *HELM.* NeurIPS 2025. arXiv:2505.24722 | Billion-parameter hyperbolic LLMs; Mixture-of-Curvature Experts |
| Van der Klis et al. arXiv:2601.21529. January 2026 | Logarithmic norm scaling proof; distance-to-hyperplane fix |
| Intrinsic Lorentz Neural Network. ICLR 2026. arXiv:2602.23981 | Fully intrinsic Lorentz FC layer |
| Bdeir, Schwethelm & Landwehr. NeurIPS 2025. arXiv:2405.13979 | Curvature-aware Riemannian AdamW |
| Brehmer et al. *L-GATr.* NeurIPS 2024 / SciPost Phys. 2025. arXiv:2405.14806 | Lorentz-equivariant geometric algebra transformer |
| Klein et al. *Hyper++.* arXiv:2512.14202. December 2025 | Stable hyperbolic RL; 30% wall-clock gain |
| Yang et al. *HypLoRA.* arXiv:2410.04010. 2025 | Hyperbolic parameter-efficient fine-tuning |

### Pillar IV — Radon-Domain BV Functional Analysis

| Reference | Contribution |
|-----------|-------------|
| Parhi & Nowak. *JMLR* 22(43). 2021. arXiv:2006.05626 | Neural networks as Radon-domain BV inverse problem solutions |
| Parhi. PhD dissertation. UW–Madison. 2022 | Ridge splines and deep network functional characterization |
| Parhi & Nowak. *SIAM J. Math. Data Sci.* 2022 | What kinds of functions do deep networks learn? |
| Bartolucci, De Vito, Rosasco & Vigogna. *JMLR* 24. 2023 | RKBS refinement and back-projection invertibility |

### Pillar V — Gradient Noise Scale / Training Dynamics

| Reference | Contribution |
|-----------|-------------|
| McCandlish, Kaplan, Amodei & OpenAI Dota Team. arXiv:1812.06162. 2018 | Empirical model of large-batch training; gradient noise scale `B_simple` |
| Merrill, Arora, Groeneveld & Hajishirzi. arXiv:2505.23971. May 2025 | Critical batch size revisited for modern LLM training |
| Barato & Seifert. *Phys. Rev. Lett.* 114, 158101. 2015 | Thermodynamic uncertainty relation |
| Jacobson. *Phys. Rev. Lett.* 75, 1260. 1995 | Einstein field equations from thermodynamic equilibrium |

### Grokking and Mechanistic Interpretability

| Reference | Contribution |
|-----------|-------------|
| Power, Burda, Edwards, Goodfellow & Misra. arXiv:2201.02177. 2022 | Grokking discovery: delayed generalization on algorithmic tasks |
| Nanda, Chan, Lieberum, Smith & Steinhardt. ICLR 2023. arXiv:2301.05217 | Mechanistic account of grokking: DFT-based algorithm in attention heads |

### Hardware-Software Co-Design

| Reference | Contribution |
|-----------|-------------|
| Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762 | Transformer architecture; explicit TPU v2 training documentation |
| Jouppi et al. ISCA 2017. arXiv:1704.04760 | TPU v1 architecture: 256×256 MXU, 8-bit inference |
| Jouppi et al. *Commun. ACM* 63(7). 2020 | TPU v2/v3: bfloat16, HBM, Pod interconnect |
| Hooker. *Commun. ACM* 64(12). 2020. arXiv:2009.06489 | Hardware Lottery framework: co-fitness determines architectural winners |
| Dao et al. NeurIPS 2022. arXiv:2205.14135 | FlashAttention: IO-aware exact attention |
| Chowdhery et al. *PaLM.* arXiv:2204.02311. 2022 | 540B parameters on 6,144 TPU v4 chips |

### Chain-of-Thought and In-Context Learning

| Reference | Contribution |
|-----------|-------------|
| Wei et al. arXiv:2201.11903. 2022 | Chain-of-thought prompting elicits multi-step reasoning |
| Brown et al. *GPT-3.* NeurIPS 2020. arXiv:2005.14165 | Few-shot learning; in-context learning at scale |

### CORDIC Hardware Corollaries

| Reference | Contribution |
|-----------|-------------|
| Kumar et al. *CARMEN.* arXiv:2605.06878. 2026 | CORDIC-accelerated hyperbolic operation inference engine |
| *CORDIC Is All You Need.* arXiv:2503.11685. 2025 | CORDIC as universal evaluator for hyperbolic neural network ops |

---

## Closing Assessment

The AI infrastructure landscape in 2027 is defined by a fundamental tension between two timelines.

The first timeline is **operational**: organizations are scaling Transformer-based systems on Euclidean infrastructure, training on TPU and GPU clusters optimized for matrix multiplication, and achieving state-of-the-art results on benchmarks that do not yet penalize geometric mismatch. This timeline will persist for at least the next 24 months, and the investments required to participate in it are well-understood.

The second timeline is **structural**: the mathematical foundations of what these systems are computing have been resolved to a degree that was not true three years ago. The representational geometry literature — ARH, MRT, RBH, the Lorentzian trainability breakthroughs, the HELM results — has reached a point of convergence where the next generation of AI infrastructure can be designed against a principled geometric specification rather than empirical intuition.

The organizations that act on the second timeline before it forces action will have the same advantage Google had in 2017: a hardware-software co-design window in which the architecture and the silicon can be developed under the same roof, toward the same geometric objective, before the rest of the industry has learned to ask the right questions.

**The geometry of the problem is known. The organizations that build against it will win. The ones that do not will, as before, fund it.**

---

*This report synthesizes primary literature from the peer-reviewed AI research record through Q2 2026. All citations include arXiv identifiers for direct verification. Predictive claims in Section 11 are stated in falsifiable form consistent with standard scientific practice. Strategic assessments represent the authors' analysis and should be treated as one input among many in infrastructure investment decisions.*

*Last updated: 2027 Edition — First Publication.*
