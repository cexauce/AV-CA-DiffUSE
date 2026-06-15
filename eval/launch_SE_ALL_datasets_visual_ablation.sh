#!/usr/bin/env bash

source ~/.bashrc
conda activate fastudiffse

set -e
set -o pipefail

RUN_ID="av_diffse_late_fusion_avhubert_icp52_6M"
DATASETS=("TCD-DEMAND") # "TCD-TIMIT-small" "LRS3-NTCD")
#DATASETS=("LRS3-NTCD")

TOTAL_SEG=10
EVAL_DIR="eval/cloned_scripts_visual_ablation/$RUN_ID"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}


# -------------------------
# Check checkpoint
# -------------------------
CKPT="logs/$RUN_ID/last.ckpt"
if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found, aborting eval"
  exit 1
fi

# -------------------------
# Clone eval scripts
# -------------------------
log "Cloning eval scripts"

python clone_eval_scripts.py \
  --run_id "$RUN_ID" \
  --datasets "${DATASETS[@]}" \
  --out_dir eval/cloned_scripts_visual_ablation

# -------------------------
# Evaluation
# -------------------------
ITOTAL_SEG=$((TOTAL_SEG - 1))

for dataset in "${DATASETS[@]}"; do
  log "Evaluating dataset: $dataset"

  SINGLE_SEG_SCRIPT="$EVAL_DIR/single_seg_launch_SE_${RUN_ID}_${dataset}.sh"

  if [ ! -f "$SINGLE_SEG_SCRIPT" ]; then
    echo "Missing eval script: $SINGLE_SEG_SCRIPT"
    exit 1
  fi

  for i in $(seq 0 $ITOTAL_SEG); do
    "$SINGLE_SEG_SCRIPT" "$i" "$TOTAL_SEG"
  done
done


