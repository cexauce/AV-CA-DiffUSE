#!/bin/bash

source ~/.bashrc

conda activate fastudiffse


# Set variables

DATASET="TCD-DEMAND"
# DATASET="TCD-DEMAND-LowSNRs"
# DATASET="TCD-TIMIT" # 1350 files
# DATASET="TCD-TIMIT-small" # 540 files
# DATASET="TCD-TIMIT-LowSNRs"
# DATASET="LRS3-NTCD"

#VISUAL_ABLATION="yes"

SEGMENT=$1
TOTAL_SEGMENTS=$2

# CKPT_PATH="path_to_audio_only_model"
# CKPT_PATH="./checkpoints/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec.ckpt"
# CKPT_PATH="./logs/av_tcd_speech_modeling_cogenav_6M_enc_dec/last.ckpt"
# CKPT_PATH="./logs/av_tcd_speech_modeling_cogenav_6M_early_late/last.ckpt"
# CKPT_PATH="./logs/av_tcd_speech_modeling_cogenav_6M_early_late_2heads/last.ckpt"
#CKPT_PATH="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo/mydiffuse/logs/GTA_aonly_tcd_speech_modeling_default_6M/last.ckpt"

CKPT_PATH="./logs/av_diffse_late_fusion_avhubert_icp52_6M_warmup_100_beta0_1_alpha_t_step_unweighted_siglip_pretrained_2_ep389/last.ckpt"
# CKPT_PATH="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/jayilo/mydiffuse/logs/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_6M_enc_dec/epoch=199-last.ckpt"
#CKPT_PATH="/group_storage/corpus/audio_visual/temp_repo/epoch=epoch=199.ckpt"

#CKPT_PATH="./logs/av_diffse_late_fusion_avhubert_icp52_6M/last.ckpt"  # baseline

#CKPT_PATH="./logs/av_diffse_late_fusion_avhubert_icp52_6M_infoNCE_trainable_audio_enc_averaged_alpha/av_diffse_late_fusion_avhubert_icp52_6M_infoNCE_trainable_audio_enc_warmup_100_alpha0_3000/last.ckpt"  # best model 
SAVE_ROOT="./eval/results" 

if [ "$DATASET" = "WSJ0" ]; then
    DATA_DIR="./eval/wsj_test.json"
elif [ "$DATASET" = "TCD-TIMIT-small" ]; then
    DATA_DIR="/group_storage/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_540.pkl"
elif [ "$DATASET" = "TCD-TIMIT" ]; then
    DATA_DIR="/group_storage/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_5.pkl"
elif [ "$DATASET" = "TCD-DEMAND" ]; then
    DATA_DIR="/group_storage/corpus/source_separation/ICP52/TCD_DEMAND/new_tcd_demand_test.pkl"
elif [ "$DATASET" = "LRS3-NTCD" ]; then
    DATA_DIR="/group_storage/corpus/source_separation/ICP52/LRS3_NTCD/new_lrs3_ntcd_test.pkl"
elif [ "$DATASET" = "TCD-DEMAND-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/tcd_demand_test_low_snrs.pkl"
elif [ "$DATASET" = "LRS3-DEMAND-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/lrs3_demand_test_low_snrs.pkl"
elif [ "$DATASET" = "LRS3-NTCD-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/lrs3_ntcd_test_low_snrs.pkl"
elif [ "$DATASET" = "VB" ]; then
    DATA_DIR="/group_storage/corpus/source_separation/VoiceBankDEMAND/vb_dmd.json"
else
    echo "Unknown dataset: $DATASET"
    exit 1  # Exit with an error if the dataset is not recognized
fi


ALGO_TYPE="fudiffse_v2" #"udiffse"   




divide_s0hat="no"
set_v_to_zero="no"


if test "$ALGO_TYPE" = "fudiffse"
then

    TAG="fudiffuse_bs4"       
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5


elif test "$ALGO_TYPE" = "udiffse"
then

    TAG="udiffuse"        
    NUM_E=30
    NUM_EM=5
    NBATCH=4
    STARTSTEP=0
    LMBD=1.5

elif test "$ALGO_TYPE" = "fudiffse_v2"
then

    TAG="fudiffse_v2"        
    NUM_E=30
    NUM_EM=1
    NBATCH=4
    STARTSTEP=0
    LMBD=1.75

else 
    echo "NOT AVAILABLE ALGO"
    break


fi


# Run command
python eval/evaluation.py \
    --dataset "$DATASET" \
    --segment "$SEGMENT" \
    --num_segments "$TOTAL_SEGMENTS" \
    --ckpt_path "$CKPT_PATH" \
    --algo_type "$ALGO_TYPE" \
    --tag "$TAG" \
    --data_dir "$DATA_DIR" \
    --save_root "$SAVE_ROOT" \
    --num_E "$NUM_E" \
    --num_EM "$NUM_EM" \
    --nbatch "$NBATCH" \
    --divide_s0hat "$divide_s0hat" \
    --set_v_to_zero "$set_v_to_zero" \
    --lambda "$LMBD" \
    --visual_ablation "$VISUAL_ABLATION"