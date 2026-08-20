#!/bin/bash
#SBATCH --job-name=rolecompress
#SBATCH --nodes=1
#SBATCH --gres=gpu:8              # 8x A6000
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# ------------------------------------------------------------------
# Usage: sbatch scripts/launch_slurm.sbatch <STAGE>
#   STAGE = prepare | labels | calibrate | router | lora | eval
# Edit the paths below for your cluster.
# ------------------------------------------------------------------
set -euo pipefail
STAGE=${1:-eval}
ROOT=$(pwd)
DATA=/data/rolecompress          # working dir for artifacts
MANIFEST=$DATA/manifest.jsonl    # {"video_id","path","captions"?,"qa"?}
MODEL=Qwen/Qwen3-VL-8B-Instruct   # 4B for fast iteration; 30B-A3B(MoE)/32B for strength
export PYTHONPATH=$ROOT:$PYTHONPATH
mkdir -p $DATA logs

case $STAGE in
  prepare)  # Stage 1: ASR + segmentation + probes + router features (8-way sharded)
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python scripts/01_prepare_data.py \
        --manifest $MANIFEST --out $DATA --model_id $MODEL \
        --make_feats --shard $i/8 &
    done; wait
    cat $DATA/probes.shard*of8.jsonl > $DATA/probes.jsonl
    ;;

  labels)   # Stage 2: score 3 frozen passes per probe (8-way sharded) -> margins
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python scripts/02_build_pid_labels.py \
        --probes $DATA/probes.jsonl --data_root $DATA --manifest $MANIFEST \
        --out $DATA/labels.jsonl --model_id $MODEL --shard $i/8 &
    done; wait
    cat $DATA/labels.margins.shard*of8.jsonl > $DATA/margins.jsonl
    ;;

  calibrate)  # Stage 2b: calibrate thresholds on merged margins -> labels.jsonl
    python scripts/02_build_pid_labels.py --calibrate_only \
      --probes $DATA/margins.jsonl --data_root $DATA --manifest $MANIFEST \
      --out $DATA/labels.jsonl
    ;;

  router)   # Stage 3: train the RoleRouter (1 GPU is enough)
    python scripts/03_train_router.py \
      --labels $DATA/labels.train.jsonl --val_labels $DATA/labels.val.jsonl \
      --feats_dir $DATA/seg_feats --out $DATA/router.pt
      # d_visual/d_text auto-inferred from cached features (= llm_hidden)
    ;;

  lora)     # Stage 4: LoRA adaptation (torchrun across 8 GPUs)
    torchrun --nproc_per_node=8 scripts/04_train_lora.py \
      --qa_train $DATA/qa_train.jsonl --manifest $MANIFEST \
      --router_ckpt $DATA/router.pt --out $DATA/lora_adapter --model_id $MODEL
    ;;

  eval)     # Stage 5: sweep policies x budgets on a benchmark
    for POL in uniform remo rolecompress oracle_role; do
      for NH in 2 4 6 8; do
        python scripts/05_eval.py --qa_eval $DATA/videomme_long.jsonl --manifest $MANIFEST \
          --policy $POL --router_ckpt $DATA/router.pt --lora $DATA/lora_adapter \
          --n_high $NH --uniform_frames $NH --tag_synergy \
          --out $DATA/results/${POL}_nh${NH}.jsonl
      done
    done
    python scripts/plot_pareto.py --results_dir $DATA/results --out $DATA/pareto.png
    ;;

  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "stage $STAGE done"
