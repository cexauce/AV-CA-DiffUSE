#!/usr/bin/env bash

source ~/.bashrc
conda activate fastudiffse

set -e
set -o pipefail

RUN_ID="av_diffse_late_fusion_avhubert_icp52_6M_infoNCE_trainable_audio_enc_warmup_50_beta0_5000_dummy"
DATASETS=("TCD_DEMAND") # "TCD_TIMIT_SMALL" "LRS3_NTCD")

TOTAL_SEG=10
EVAL_DIR="eval/cloned_scripts/$RUN_ID"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# -------------------------
# Training
# -------------------------
log "Starting training for $RUN_ID"

python train.py \
  --transform_type exponent \
  --format tcd-timit \
  --batch_size 2 \
  --gpus 1 \
  --dummy \
  --regularization_warmup_epochs 50 \
  --regularization_beta0 5000 \
  --vfeat_processing_order cut_extract \
  --video_feature_type avhubert \
  --backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
  --fusion concat_attn_masking_light \
  --no_project_video_feature \
  --p 0.0 \
  --fusion_level enc_dec \
  --run_id "$RUN_ID"

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
  --out_dir eval/cloned_scripts

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


