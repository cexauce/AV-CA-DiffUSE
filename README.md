# AV-CA-DiffUSE : Audio-visual Contrastive Alignment for Diffusion-based Visual-conditioned Speech Enhancement


Official PyTorch implementation of:

> Colombe Mboungou, Mostafa Sadeghi, Jean-Eudes Ayilo, Romain Serizel "Audio-visual Contrastive Alignment for Diffusion-based Visual-conditioned Speech Enhancement" accepted at Interspeech 2026.


## Installation

After cloning this repository, create a virtual environment and install the package dependencies:

```bash
cd AV-CA-DiffUSE

conda create -n fastudiffse python=3.8.16

conda activate fastudiffse

pip install -r requirements.txt
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

- Training the conditional audio-visual model with audio-visual contrastive alignment used in the paper

```bash
python train.py \
	--transform_type exponent \
	--format tcd-timit \
	--batch_size 8 \
	--gpus 2 \
	--regularization_warmup_epochs 100 \
	--regularization_beta0 1000 \
	--alpha_t_decay step \
	--vfeat_processing_order cut_extract \
	--video_feature_type avhubert \
	--backbone ncsnpp_continueconcat_attn_masking_noising_av_6m \
	--fusion concat_attn_masking_light \
	--no_project_video_feature \
	--p 0.0 \
	--fusion_level enc_dec \
	--run_id av_diffse_late_fusion_avhubert_icp52_6M_warmup_100_beta0_1000_alpha_t_step_unweighted
```

To run a full size (unconditional) model for the audio-only case, similar to the NCSN++ 65M paramaters based, just change `ncsnpp28M` into `ncsnpp`

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
	--run_id id_for_your_auido_visual_model
```


To run the full size model for av, just change `ncsnpp_continueconcat_attn_masking_noising_av_28m` into `ncsnpp_continueconcat_attn_masking_noising`


- To choose your fusion method, use the fusion input argument :
	- 'early' for early fusion
	- 'concat_attn_masking_light' for late fusion
	- 'concat_attn_masking_light_early_late' for mixed fusion (both early and late fusion using the same visual embedding you specified)


## Pretrained checkpoints
 
The checkpoint of the audio-only and the audiovisual diffusion models trained on TCD-TIMIT clean speech in the paper are available following these links:
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


## Demo

A demo notebook is available at [./demo.ipynb](./demo.ipynb) . This notebook provides a demonstration of sampling from clean speech prior learned via a diffusion-based generative model unconditionned or conditionned on lip video, followed by speech enhancement of a test noisy speech signal.


## Acknowledgements

This repository is mainly derived from  [fast_UdiffSE](https://github.com/jeaneudesAyilo/fast_UdiffSE).


## Bibtex

```bibtex
The article will be published soon this summer on arxiv, Interspeech and HAL. 
```