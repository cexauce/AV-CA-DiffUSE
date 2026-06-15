#!/bin/bash

source ~/.bashrc

conda activate fastudiffse

# Set variables
 
ENHANCED_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/fast_UdiffSE_fusion/eval/results/av_diffse_late_fusion_avhubert_icp52_6M/TCD-DEMAND/fudiffse_v2" 
#ENHANCED_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/fast_UdiffSE_fusion/eval/results/av_diffse_early_fusion_avhubert_6M_/av_diffse_early_fusion_avhubert_6M/TCD-DEMAND/fudiffse_v2/1.75/fudiffse_v2"
# /group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/GTA_aonly_tcd_speech_modeling_default_28M/TCD-DEMAND/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/GTA_aonly_tcd_speech_modeling_default_28M/TCD-DEMAND/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/TCD-DEMAND/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/TCD-DEMAND/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/flow_avse_enh_tcd_demand_best_sisdr/first_stage"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/TCD-DEMAND/flow_avse_enh_tcd_demand_best_sisdr/second_stage"


# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/GTA_aonly_tcd_speech_modeling_default_28M/LRS3-NTCD/fudiffse/fudiffuse_bs4"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/GTA_aonly_tcd_speech_modeling_default_28M/LRS3-NTCD/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/LRS3-NTCD/fudiffse/fudiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec/LRS3-NTCD/udiffse/udiffuse"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/flow_avse_enh_lrs3_ntcd_best_sisdr/first_stage"
# ENHANCED_DIR="/group_calcul/calcul/users/jayilo/ICP_52_results/grat/LRS3-NTCD/flow_avse_enh_lrs3_ntcd_best_sisdr/second_stage"


DATASET="TCD-DEMAND"
# DATASET="LRS3-NTCD"
# DATASET="TCD-TIMIT-small" # 540 files

if [ "$DATASET" = "WSJ0" ]; then
    DATA_DIR="./eval/wsj_test.json"
elif [ "$DATASET" = "TCD-TIMIT-small" ]; then
    DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_540.pkl"
elif [ "$DATASET" = "TCD-TIMIT" ]; then
    DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_5.pkl"
elif [ "$DATASET" = "TCD-DEMAND" ]; then
    DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/source_separation/ICP52/TCD_DEMAND/new_tcd_demand_test.pkl"
elif [ "$DATASET" = "LRS3-NTCD" ]; then
    DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/source_separation/ICP52/LRS3_NTCD/new_lrs3_ntcd_test.pkl"
elif [ "$DATASET" = "TCD-DEMAND-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/tcd_demand_test_low_snrs.pkl"
elif [ "$DATASET" = "LRS3-DEMAND-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/lrs3_demand_test_low_snrs.pkl"
elif [ "$DATASET" = "LRS3-NTCD-LowSNRs" ]; then
    DATA_DIR="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/msadeghi/pytorch/SE-Diffusion/LipVoicer/eval/lrs3_ntcd_test_low_snrs.pkl"
elif [ "$DATASET" = "VB" ]; then
    DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/source_separation/VoiceBankDEMAND/vb_dmd.json"
else
    echo "Unknown dataset: $DATASET"
    exit 1  # Exit with an error if the dataset is not recognized
fi
echo "$ENHANCED_DIR"
echo "$DATASET"
echo "$DATA_DIR"


# Run command, for LRS3-NTCD do not apply dnn_mos (due to the short length of some files)
python eval/statistics/compute_metrics.py \
    --enhanced_dir "$ENHANCED_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$ENHANCED_DIR" \
    --dataset  "$DATASET" \
    --input_metrics
    # --dnn_mos
    