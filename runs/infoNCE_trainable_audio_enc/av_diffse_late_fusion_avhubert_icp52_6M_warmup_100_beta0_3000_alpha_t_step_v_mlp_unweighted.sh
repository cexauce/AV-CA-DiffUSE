
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
	--format tcd-timit \
	--batch_size 8 \
	--gpus 2 \
	--no_wandb \
	--regularization_warmup_epochs 100 \
	--regularization_beta0 3000 \
	--mlp_before_fusion \
	--alpha_t_decay step \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
	--fusion concat_attn_masking_light \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id av_diffse_late_fusion_avhubert_icp52_6M_warmup_100_alpha0_3000_alpha_t_step_vmlp_unweighted \
	#	--dummy \