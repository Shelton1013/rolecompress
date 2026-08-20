# RoleCompress

**Compressing long-video tokens by cross-modal information role.**

A frozen video-LLM (**Qwen3-VL-8B**, 2026; Qwen2.5-VL / LLaVA-Video also supported) + a
lightweight **RoleRouter** that classifies each video segment's information role — cross-modal **Redundant** (text/ASR covers it → drop
visual), unimodal **Unique-Visual** (keep sparse), cross-modal **Synergistic** (keep dense)
— trained with **self-supervised labels from single- vs joint-modality head disagreement**,
then allocates the frame/token budget by role. Query-agnostic; one router pass per video,
reused across turns. Only the RoleRouter (~5–15M) + LoRA are trained; the backbone is frozen.

See `EXPERIMENT_PLAN.md` for the full protocol, baselines, ablations, and the novelty
positioning vs the nearest prior work (ReMo / Omni2LoRA / SPICE / EchoingPixels).

## Why it's novel (one paragraph)
Existing long-video efficiency methods allocate the visual budget by *saliency*,
*query-relevance*, or *modality balance*, and therefore preferentially destroy the small
but decisive **cross-modal synergistic** information. RoleCompress is the first to (a) use
the full PID trichotomy (redundant/unique/**synergy**) as the *allocation policy*, (b) with
a **synergy-preserving** branch, (c) trained from **self-supervised head-disagreement**
labels, (d) for long video on a frozen backbone. The novelty survey (25 papers, in
`../longvideo_survey/`) confirms no direct collision.

## Hardware
Designed for **8× A6000 (384 GB)**. Everything is inference-heavy + small-module training;
default is LoRA (no 7B optimizer states). Full-FT is optional (FSDP + ZeRO-3 offload).

## Backbone (2026)
Default: **`Qwen/Qwen3-VL-8B-Instruct`** (needs `transformers>=4.57`; if not yet released,
`pip install "git+https://github.com/huggingface/transformers"`). The wrapper is
family-aware — switch by `--model_id`:
- `Qwen/Qwen3-VL-4B-Instruct` — fast iteration (llm_hidden 2560)
- `Qwen/Qwen3-VL-8B-Instruct` — default (llm_hidden 4096)
- `Qwen/Qwen3-VL-30B-A3B-Instruct` (MoE) / `Qwen/Qwen3-VL-32B-Instruct` — stronger
- `Qwen/Qwen2.5-VL-7B-Instruct` (also install `qwen-vl-utils`) or a LLaVA-Video ckpt — for backbone-transfer ablations.

Router feature dims (`d_visual`/`d_text`) are **auto-inferred** from the cached features
(= the backbone's `llm_hidden`), so you never hardcode them.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Qwen3-VL: transformers>=4.57 (or git main). flash-attn optional; attn_impl defaults to "sdpa".
```

## Validate the logic locally FIRST (no GPU)
The novel logic (role assignment, budget allocation, soft targets, metrics, router shapes)
is unit-tested and GPU-free. Run this before burning server time:
```bash
pip install pytest
PYTHONPATH=. pytest -q tests/
```

## Data manifest
One jsonl line per video:
```json
{"video_id": "v001", "path": "/data/videos/v001.mp4",
 "captions": [{"start":0,"end":6,"text":"..."}],   // optional, for probe generation
 "qa": [{"question":"...","choices":[".."],"answer":"A","start":12}]  // optional
}
```
For eval benchmarks (Video-MME long / MLVU / LongVideoBench / EgoSchema), convert to:
```json
{"video_id":"v001","question":"...","choices":["..",".."],"answer":"B"}
```
(a small converter per benchmark is the only glue you write; the fields above are all the
code needs.)

## Pipeline (see scripts/, or scripts/launch_slurm.sbatch)
```
1. prepare : ASR + segmentation + segment-probes + cached router features   (01_prepare_data.py)
2. labels  : 3 frozen passes/probe -> margins                               (02_build_pid_labels.py)
   calibrate: thresholds on merged margins -> labels.jsonl (soft+hard roles) (02 ... --calibrate_only)
   (split labels.jsonl into labels.train/val.jsonl by video)
3. router  : train RoleRouter on cached features + soft role targets        (03_train_router.py)
4. lora    : LoRA-adapt backbone to the role-allocated token stream         (04_train_lora.py)
5. eval    : sweep policies x budgets; synergy subset; Pareto + crossover   (05_eval.py, plot_pareto.py)
```
Slurm one-liners:
```bash
sbatch scripts/launch_slurm.sbatch prepare
sbatch scripts/launch_slurm.sbatch labels
sbatch scripts/launch_slurm.sbatch calibrate
sbatch scripts/launch_slurm.sbatch router
sbatch scripts/launch_slurm.sbatch lora
sbatch scripts/launch_slurm.sbatch eval
```

## Baselines (all at matched visual budget, same backbone/LoRA)
`uniform` · `remo` (cross-modal redundancy drop, our re-impl) · `random_role` · `oracle_role`
(upper bound) · `rolecompress` (ours). FastV / token-merge / query-aware selectors plug in
as extra `--policy` values (stubs indicated in `05_eval.py::policy_roles`).

## The two headline results
1. **Accuracy–budget Pareto** (all benchmarks): RoleCompress dominates uniform/FastV/ReMo.
2. **Synergy-subset crossover**: on questions where a frozen text-only and vision-only pass
   both fail but the joint passes, ReMo-style redundancy compression collapses as budget
   shrinks while RoleCompress holds — the acceptance-securing figure.

## Repo layout
```
rolecompress/            core package
  roles.py               roles, budget policy, allocation  (unit-tested)
  pid_labels.py          self-supervised role labels from head margins
  router.py              RoleRouter + loss
  backbone.py            family-aware VL wrapper (Qwen3-VL default): score_probe / features / role-allocated QA  [VERIFY tags]
  segment.py             windowing + decord frame reader
  asr.py                 faster-whisper + segment alignment
  data.py                jsonl IO + router dataset
  metrics.py             accuracy / budget / FLOPs / synergy subset  (unit-tested)
scripts/                 01..05 pipeline + plot + slurm launcher
configs/default.yaml     defaults
tests/                   GPU-free logic tests — run first
EXPERIMENT_PLAN.md       full protocol
```

## `[VERIFY]` markers
`backbone.py` contains a handful of `[VERIFY]` comments where behavior depends on the exact
`transformers` / model version (processor kwargs, visual tower call, `get_rope_index` for the
optional token-merge path, LoRA target-module names). The **frame-budget path used in all
main experiments does not touch positional encodings** and is robust; check the markers only
if you change backbone or transformers major version.
```
