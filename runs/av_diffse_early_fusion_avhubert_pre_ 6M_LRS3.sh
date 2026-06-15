
source ~/.bashrc
conda activate fastudiffse


# python train.py \
# 	--transform_type exponent \
# 	--format tcd-timit \
# 	--batch_size 8 \
# 	--gpus 2 \
# 	--vfeat_processing_order cut_extract \
# 	--video_feature_type cogenav \
# 	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
# 	--fusion concat_attn_masking_light \
# 	--no_project_video_feature \
# 	--p 0.0 \
# 	--fusion_level enc_dec \
# 	--run_id av_tcd_speech_modeling_cogenav_6M_enc_dec \
# 	--no_wandb

python train.py \
	--transform_type exponent \
	--format lrs3 \
	--batch_size 8 \
	--gpus 2 \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert_pre \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
	--fusion early \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id av_diffse_early_fusion_avhubert_6M_LRS3 \
	--no_wandb