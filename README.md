## Overview
Official pytorch implementation of the paper AV-CA-DiffUSE : Audio-visual Contrastive Alignment for Diffusion-based Visual-conditioned Speech Enhancement.
AV-CA-DiffUSE extends AV-UDiffSE by introducing an explicit
audio-visual contrastive alignment objective during training.

The contrastive loss encourages stronger correspondence between
audio and visual representations while preserving the diffusion-based
speech enhancement framework. The proposed approach improves
interference suppression and reconstruction quality under both matched
and mismatched conditions without modifying the inference architecture.

## Installation

After cloning this repository, create a virtual environment and install the package dependencies:

```bash
cd AV-CA-DiffUSE

conda create -n fastudiffse python=3.8.16

conda activate fastudiffse

pip install -r requirements.txt
```
## Repository Structure

```
├── data_demo/      	Data for the demo notebook
├── eval/               Evaluation scripts and metrics
├── preprocessing/      Data preparation scripts
├── runs/      			Training scripts
├── sgmse/              Diffusion and fusion models
├── demo.ipynb          Demonstration notebook
├── train.py            Training entry point
└── requirements.txt    Python dependencies
```

## Data

The video of lips ROI (88 $\times$ 88 and 112 $\times$ 112) for TCD-TIMIT (train/valid/test) and LRS3 test set are provided following the links: 

- [TCD-TIMIT (train/valid/test) and LRS3-test lips video in 88 $\times$ 88 format](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/CROPPED_MOUTH_ldmark_48_68_size_88_88.tar.gz)   
- [TCD-TIMIT (train/valid/test) and LRS3-test lips video in 112 $\times$ 112](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/CROPPED_MOUTH_ldmark_28_68_size_112_112.tar.gz)

The created noisy speech are available at:

- [TCD-DEMAND](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/TCD_DEMAND.tar.gz)
- [LRS3-NTCD](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/LRS3_NTCD.tar.gz)

For the clean speech, please consider getting the original clean speech video of TCD-TIMIT at [https://sigmedia.tcd.ie/tcd_timit_db/volunteers](https://sigmedia.tcd.ie/tcd_timit_db/volunteers), and for LRS3: [https://mmai.io/datasets/lip_reading/](https://mmai.io/datasets/lip_reading/) (LRS3 might have availabity issue).
Regarding NTCD-TIMIT, please visit: [https://zenodo.org/records/1172064](https://zenodo.org/records/1172064)
It could be interesting to take a look at [./preprocessing](./preprocessing) to see how this noisy speech data have been generated.

## Training

- Training the conditional audio-visual model with audio-visual contrastive alignment speech enhancement model introduced in the paper

```bash
python train.py \
	--transform_type exponent \ 			# transformation applied to the audio input STFT 
	--format tcd-timit \					# dataset
	--batch_size 8 \						# batchsize		
	--gpus 2 \								# nb of gpus	
	--regularization_warmup_epochs 100 \	# nb of warmup epochs before the contrastive loss is applied 
	--regularization_beta0 1000 \			# value of weight of the contrastive loss beta at the last epoch
	--perceptron_before_fusion \			# adds linear layer before fusion
	--alpha_t_decay step \					# how the contrastive loss weight alpha decreases through noising steps
	--vfeat_processing_order cut_extract \	# how visual input frames are cut to match the nb audio of frames
	--video_feature_type avhubert \			# the type of visual embeddings used to encode visual input
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \	# neural network architecture 
	--fusion concat_attn_masking_light \							# fusion
	--no_project_video_feature \									# whether a projection layer is added after fusion
	--p 0.0 \														# audio masking probability
	--fusion_level enc_dec \										# level at which fusion is performed
	--run_id av_diffse_late_fusion_avhubert_icp52_6M_warmup_100_beta0_1000_alpha_t_step_unweighted	# name of the checkpoint folder 
```
You can use `--dummy` for faster debugging




- Training the conditional audio-visual model used in the paper for AV-UDiffSE and AV-UDiffSE+ (Table 1 of the paper)


```bash
python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 8 \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_28m \
	--fusion concat_attn_masking_light \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id av_diffse_late_fusion_avhubert_icp52_6M
```





## Pretrained checkpoints

 
The checkpoint of the audio-only and the audiovisual diffusion models trained on TCD-TIMIT clean speech in the paper are available following these links:
- [audiovisual-contrastive-alignment-linear-layer](https://drive.google.com/file/d/1MaDukx37NigLupGUQ0QlzY7D75L0h3yl/view?usp=drive_link)
- [audiovisual-contrastive-alignment](https://drive.google.com/file/d/1kIg-4Vv26Y2bkW_Pq4BHFySytZ7qD2is/view?usp=drive_link)
- [audio-only ckpt](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/aonly_tcd_speech_modeling_default_28M.ckpt)
- [audiovisual ckpt](https://huggingface.co/jeaneudesAyilo/files_for_fast_UdiffSE/resolve/main/av_tcd_speech_modeling_concat_attn_masking_light_avhubert_p0_28M_enc_dec.ckpt)


## Evaluation

Run enhancement by providing the suitable information in the file [./eval/SE_eval.sh](./eval/SE_eval.sh). 

- If one is working on a single gpu, consider writing SEGMENT=0 and TOTAL_SEGMENTS=1 in the `./eval/SE_eval.sh` file, then run 

```bash
bash launch_SE_ALL
```

- If one is working with OAR batch scheduler for HPC clusters, provide the suitable information in the file [./eval/SE_eval.sh](./eval/SE_eval.sh) do not change SEGMENT=$1 and TOTAL_SEGMENTS=$2. Consider providing the names of clusters in [./eval/single_seg_launch_SE.sh](./eval/single_seg_launch_SE.sh). By doing so, you allow the test set to be divided into segments that will be individually proceeded by a different gpu on the clusters or nodes. Then run 

```bash
bash launch_SE_ALL.sh
```


- Compute metrics
Metrics are computed directly after finishing the enhancement of test files in [./eval/evaluation.py](./eval/evaluation.py). We can compute the metrics independently by running: 

```bash
python eval/statistics/compute_metrics.py \
	--enhanced_dir your_enhanced_dir  
	--data_dir path_to_pickle_file_of_test_files_list \
	--save_dir your_enhanced_dir  \
	--dataset dataset_name \
	--dnn_mos
```
- In the above command, consider removing the `dnn_mos` argument if MOS metrics are not needed, or if the test set contains short audio files. We can add `--input_metrics` to compute metrics of the non-enhanced noisy speech.



## Acknowledgements

This repository is mainly derived from  [fast_UdiffSE](https://github.com/jeaneudesAyilo/fast_UdiffSE).


## Citation
If you find this repository useful, please cite:
```bibtex
@inproceedings{Mboungou2026av, title={Audio-visual Contrastive Alignment for Diffusion-based Visual-conditioned Speech Enhancement}, author={Mboungou, Colombe and Sadeghi, Mostafa and Ayilo, Jean-Eude and Serizel, Romai}, booktitle={INTERSPEECH}, year={2026} }

```