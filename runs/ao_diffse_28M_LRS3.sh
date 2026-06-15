
source ~/.bashrc
conda activate fastudiffse


python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 2 \
	--gpus 1 \
	--audio_only \
	--vfeat_processing_order default \
	--backbone ncsnpp_continueconcat_attn_masking_noising \
	--dummy \
	--run_id aonly_tcd_speech_modeling_default_28M_LRS3 \
	--no_wandb
	#	--batch_size 8 \--gpus 2 \