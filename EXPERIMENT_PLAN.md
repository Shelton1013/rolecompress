# RoleCompress — Experiment Plan

**Method name (working):** RoleCompress — *Compressing long-video tokens by cross-modal information role.*

**One-line thesis.** Existing long-video efficiency methods allocate the visual budget by *saliency*, *query-relevance*, or *modality balance* — and therefore preferentially destroy the small but decisive **cross-modal synergistic** information. We instead classify each video segment's information **role** — cross-modal **Redundant** (text/ASR already conveys it → drop visual), unimodal **Unique-Visual** (keep, sparse), cross-modal **Synergistic** (keep dense) — using **self-supervised labels from single- vs joint-modality head disagreement**, and allocate the frame/token budget by role. Query-agnostic, one router pass per video, reused across turns; frozen 7B backbone + LoRA.

This document is the full experimental protocol; the code repo implements every step.

---

## 1. Positioning & novelty (what reviewers will check)

| Prior art (nearest) | What it does | Our delta |
|---|---|---|
| **ReMo** (2607.21179) | Drops visual tokens whose info appears elsewhere, via cosine similarity; text proxies | Only the *redundant→drop* half, by saliency; **no unique/synergy axis**, **no PID**, **no learned/self-sup labels** |
| **Omni2LoRA** (2608.09227) | Allocates *LoRA-rank* memory to synergistic anchors via RL reward | We allocate the **input frame/token budget**, per-segment, by explicit role; self-supervised not RL |
| **SPICE** (2606.16639) | Same single-vs-joint disagreement signal — but to order **training curriculum** | We use it to produce **per-segment role labels** that drive **inference-time compression** |
| **EchoingPixels** (2512.10324) | "Synergistic" = interdependent importance (Top-K) | Name overlap only; ours is PID R/U/S |
| PID-analysis line (Sensory PID 2606.00959, 2603.29676, 2602.15580, Liang 2302.12247) | *Diagnose* R/U/S | We *act* on R/U/S to compress; long video; a method |

**Defensible novelty = the combination:** (a) three-way role (not just redundancy) driving a frame/token budget, (b) a **synergy-preserving** branch, (c) **self-supervised role labels** from frozen-head disagreement, (d) long-video, query-agnostic, frozen backbone + LoRA.

**Killer result to secure acceptance:** on a *synergy-required* subset of long-video QA, redundancy-only / saliency compression (ReMo-style, FastV) collapses as the budget shrinks, while RoleCompress holds → a clean crossover curve.

---

## 2. Backbone & why

**Backbone choice is a fairness decision, not a "use the newest model" decision.** The
comparison must be *same-backbone, different token policy*. We surveyed the backbones used by
this literature (our baselines): **Qwen2.5-VL-7B (~14 papers, the de-facto standard),
LLaVA-Video-7B (~10), LLaVA-OneVision-7B (~6)**, plus InternVL2.5/Qwen2-VL. Crucially,
**CVPR'26 papers were submitted Nov-2025 / camera-ready Apr-2026 — before Qwen3-VL (Aug-2026)**,
so none of them use Qwen3-VL. Therefore:

- **Primary (main comparison table):** `Qwen/Qwen2.5-VL-7B-Instruct` — matches the baselines, so
  ReMo / FastV / token-merge / query-selection are re-run on the *same* backbone (only the token
  policy differs). Frozen. Fast-iteration: `Qwen/Qwen2.5-VL-3B-Instruct`.
- **Generalization row:** `llava-hf/LLaVA-Video-7B-Qwen2` — shows the method transfers across the
  second most common backbone.
- **Latest-backbone bonus row:** `Qwen/Qwen3-VL-8B-Instruct` (needs `transformers>=4.57`) — "still
  helps on the newest backbone"; a bonus, *not* the basis of the head-to-head comparison.
- **Extension (follow-up / ablation):** `Qwen3-Omni` to show native audio is a drop-in replacement
  for the ASR-text stream. Note **ReMo itself is reported on Qwen2.5-Omni** (it is audio-visual);
  we compare against our same-backbone re-implementation of its redundancy-drop, and optionally
  also run the Omni backbone to match ReMo's exact setting.
- **What we train:** the **Role Router** (~5–15M params) + **LoRA** (rank 16–32) on the LLM's attention/MLP proj. Backbone weights frozen. Fits comfortably on 8×A6000 (48 GB×8 = 384 GB): a frozen 7B + LoRA is light; the 3B variant is for fast iteration.

---

## 3. Two allocation granularities (code supports both)

1. **Frame-budget (primary, robust, runnable).** Per segment, role → number of sampled frames:
   `Redundant → 0 frames` (rely on ASR text tokens injected inline) `· Unique-V → n_low (e.g. 1)` `· Synergy → n_high (e.g. 4)`.
   Realized purely by controlling which frames go to the processor — **no positional-encoding surgery**. This is the main experimental system.
2. **Token-merge (secondary, higher compression).** Within kept frames, redundant/unique segments get bilinear-pooled visual tokens (2×/4× merge) while synergy keeps full resolution. Implemented at the input-embedding level with a re-computed `get_rope_index`. Reported as an extra Pareto point; flagged as the version-dependent path.

Budget is reported as **effective visual tokens** and **prefill FLOPs proxy** (∝ (visual_tokens)² for attention).

---

## 4. Self-supervised role labels (the training signal)

For each segment *s* of a training video with an auto- or human-provided local probe QA `(q_s, a_s)` (see §5), score three frozen heads of the *same* backbone (no extra models):

- `p_T` = P(a_s | q_s, **ASR text of s only**) — text/language head
- `p_V` = P(a_s | q_s, **frames of s only**) — vision head
- `p_J` = P(a_s | q_s, **text + frames of s**) — joint

Define margins `mT, mV, mJ` = log-likelihood of gold answer (or MCQ logit margin). Role assignment (thresholds τ on a dev split):

| Condition | Role |
|---|---|
| `mT ≥ τ_hi` (text alone suffices) | **Redundant** (drop visual) |
| `mV ≥ τ_hi` and `mT < τ_lo` (vision alone, text can't) | **Unique-Visual** |
| `mJ ≥ τ_hi` and `max(mT, mV) < τ_lo` (only joint) | **Synergistic** |
| otherwise | **Unique-Visual** (safe default — keep some visual) |

This is a **soft** target: we store the continuous vector `(mT, mV, mJ)` and a 4-way soft label (softmax over role scores derived from the margins) so the router is trained with KL, not just argmax. No human role annotation needed — labels come from the backbone's own heads.

*Sanity check to report:* agreement between self-sup roles and a small human-labeled role set (~500 segments) — expect ≥70–80% top-1; this validates the signal (mirrors how the synergy papers validate PID).

---

## 5. Data

**Training (router + LoRA):**
- Long-video instruction/QA with timestamps for local probes: **LLaVA-Video-178K** (timestamped), **Cinepile**, **ActivityNet-Captions / QA**, **NExT-QA** (train). We derive per-segment probes from timestamped captions/QA (cloze/MCQ). Target ≈ **80–150k segment-probes** across ≈ 8–15k videos (enough for a router + LoRA; not a foundation-model scale).
- Auto-probe generation: for a segment, turn its caption/subtitle into a cloze ("what is happening / what object / what is said") + 3 distractors sampled from other segments. Keep only probes that at least one head answers (filters degenerate ones).

**Evaluation (long-video QA):**
- **Video-MME** (long split), **MLVU**, **LongVideoBench**, **EgoSchema** (accuracy vs budget).
- **Cross-modal / synergy-required eval:** **Daily-Omni**, **WorldSense**, **AVSCapBench** (caption→QA), and a **self-constructed synergy subset**: from the above, keep questions where a frozen text-only pass and a frozen vision-only pass both fail but the joint passes (this *operationalizes* "synergy-required" and is the crossover-experiment testbed). Report its size and the construction protocol (this doubles as a small benchmark contribution, positioned as a *diagnostic subset*, not a new benchmark).

**ASR:** faster-whisper (large-v3, int8/float16) per video → sentence segments with timestamps; the text stream is the "cheap dense modality." For no-speech videos, ASR is empty → Redundant role is rare → method degrades gracefully to vision-only (report this).

---

## 6. Baselines (all at matched visual-token budgets)

1. **Uniform** frame sampling (the standard).
2. **FastV** (attention-based token pruning at a mid LLM layer).
3. **Token-merge / ToMe-video** (bilinear/similarity merge).
4. **Query-aware selection** (a keyframe selector, e.g. re-implement a simple conditional-MI / relevance selector).
4b. **GIFT** (2603.25072, *Global Irreplaceability Frame Targeting*) — **STRONG query-aware keyframe selector, main-table required baseline.** Reports on **Qwen2.5-VL-7B (our backbone)** → fair same-backbone comparison is possible; on MLVU 8f it lifts uniform 56.4 → **65.8**. Two structural differences we exploit, NOT raw MLVU accuracy (a dedicated query-aware selector will likely lead single-query MLVU acc — do not claim to beat it there):
   - **query-aware, per-question recompute** vs our **query-agnostic, compute-once-reuse-all** → we win on *amortized* cost when a video carries many questions (see §7 amortized-cost figure).
   - **vision-only** (no audio/ASR) → structurally cannot exploit cross-modal redundancy/synergy → we win on the AV/synergy benchmarks (Daily-Omni / WorldSense), where GIFT/AKS/FastV have no mechanism.
5. **ReMo-style redundancy drop** (cross-modal cosine similarity drop, our re-implementation) — *the key baseline*. NOTE: ReMo is **not open-sourced**; the rigorous, controlled stand-in is our own `--no_synergy` ablation (rolecompress with SYNERGISTIC→UNIQUE_VISUAL = redundancy+unique only, same pipeline/backbone), which isolates the synergy branch without depending on an external re-impl.
6. **Random role** (ablation: router replaced by random role assignment) — isolates the value of learned roles.
7. **Oracle role** (roles from the self-sup labels at test time using gold answers) — upper bound.

All share the same frozen backbone + the same LoRA (or no-LoRA variant) so the comparison is the *allocation policy*, not the backbone.

> **Positioning vs query-aware SOTA (GIFT/Q-Frame/FOCUS/AKS):** these are strong on single-query vision-centric MLVU because they see the question. Our two defensible axes are (a) **query-agnostic amortization** and (b) **cross-modal roles**. The paper's headline is NOT "top MLVU acc" but the **amortized-cost Pareto** + **synergy crossover**; on MLVU our target is *do-no-harm at a single video encode*.

---

## 7. Metrics

- **Accuracy** (per benchmark; MCQ acc / open-ended with LLM-judge where needed).
- **Visual budget:** effective visual tokens (mean), and **prefill FLOPs proxy** (attention ∝ N²).
- **Latency / TTFT** and peak memory (measured on the A6000 node).
- **Pareto:** accuracy vs visual tokens (the headline plot).
- **Amortized-cost figure (vs query-aware SOTA like GIFT):** cumulative compute (frame-encode + selection/probe passes) to answer **Q questions on the same video**, vs accuracy. Query-aware selectors (GIFT) pay their selection cost **per question** (re-encode all frames × Q); RoleCompress pays the role pass **once** and reuses. Curve crosses over as Q grows — the deployment-realistic regime (MLVU has many questions per video). This is the figure that answers "GIFT beats us on single-query MLVU acc."
- **Synergy-subset accuracy** vs budget (the crossover plot; RoleCompress ≫ redundancy-only at low budget).
- **Role stats:** distribution of R/U/S across datasets; correlation of predicted synergy fraction with a PID estimate (GPID, 2510.04417) on a subsample — ties the empirical result to theory.

---

## 8. Ablations (Section 5 of paper)

| Ablation | Question |
|---|---|
| remove Synergy branch (only R/U) | does keeping synergy actually matter? (expect big drop on synergy subset) |
| remove Redundant drop (only U/S) | where does the efficiency come from? |
| **training-free variant** (`rolecompress_tf`) | roles from per-segment answer-confidence probes (no router, no LoRA). Shows the info-role idea works with **zero training** — but it is query-dependent and costs 3 forward passes/segment; the trained Router recovers it query-agnostically and cheaply. This is the apples-to-apples row vs training-free FastV/ReMo. |
| random / oracle role | value of the learned router; headroom |
| self-sup label source: joint-head-disagreement vs attention-based vs caption-length heuristic | is the head-disagreement signal necessary? |
| soft (KL) vs hard (argmax) role targets | |
| budget levels `n_low/n_high` sweep | Pareto |
| LoRA vs frozen (no adapt) vs full-FT (if it fits) | how much adaptation is needed |
| query-agnostic reuse: same roles across N questions vs recompute per question | shows the multi-turn advantage over query-conditioned selection |
| PID estimator (GPID) as router target vs head-disagreement | theory link / cost |
| backbone transfer: Qwen2.5-VL → LLaVA-Video → Qwen2.5-Omni(+audio text) | generality |

---

## 9. Compute budget on 8×A6000 (384 GB)

| Stage | What | Est. time |
|---|---|---|
| ASR + segmentation | faster-whisper on ~15k videos | ~1–2 days (CPU/GPU mixed, parallelizable) |
| PID label build | 3 frozen-head forward passes × ~120k probes | ~1–2 days (inference only, batched, 8-GPU) |
| Router training | 5–15M params on cached features | hours |
| LoRA training | rank-16/32 on frozen 7B, ~2–3 epochs on ~120k QA | ~1–3 days |
| Eval sweep | 4 benchmarks × ~7 baselines × budget grid | ~1–2 days |

Everything is inference-heavy + small-module training → **fits 8×A6000**; no full 7B optimizer states needed. (Full-FT of 7B is the only thing near the limit; we default to LoRA and only try full-FT if time permits with FSDP + ZeRO-3 CPU offload.)

---

## 10. Timeline (aggressive, ~8 weeks to submission)

1. **W1:** data pipeline (ASR, segmentation, probe generation) + backbone wrapper + frame-budget forward path. Reproduce Uniform baseline numbers.
2. **W2:** PID self-sup labeling + human-label sanity check (500 seg). Router train + role-accuracy.
3. **W3:** LoRA train with role allocation; first Pareto vs Uniform/FastV.
4. **W4:** ReMo + query-aware + token-merge baselines (re-impl); synergy-subset construction.
5. **W5:** killer crossover experiment + oracle/random role; ablations round 1.
6. **W6:** backbone transfer (LLaVA-Video, Omni+audio) + GPID theory link.
7. **W7:** ablations round 2, latency/memory, figures.
8. **W8:** writing, rebuttal-proofing (the 4 nearest-prior distinctions), release code.

---

## 11. Risks & mitigations

- **Synergy is tiny (<2%) → small margins.** Mitigate by (a) reporting the *synergy subset* where it's concentrated, (b) framing efficiency (redundancy drop) as the main win and synergy-preservation as the *robustness* story. Both are contributions.
- **Self-sup labels noisy.** Soft KL targets + the human sanity check + oracle-role upper bound quantify the ceiling.
- **Backbone token-surgery fragility.** Default to frame-budget (no rope surgery); token-merge is optional.
- **ReMo baseline strength.** Re-implement faithfully and matched-budget; our advantage must be on synergy subset, not raw accuracy.

---

## 12. Deliverables in this repo

- `rolecompress/` package (roles, PID labels, router, backbone wrapper, data, metrics, asr, segment).
- `scripts/` numbered pipeline (prepare → labels → train router → train LoRA → eval).
- `configs/` yaml.
- `tests/` GPU-free unit tests for the novel logic (roles, budget, labels, router shapes) — **run these locally first to validate logic before the server run.**
- `scripts/launch_slurm.sbatch`, `run_all.sh`.
