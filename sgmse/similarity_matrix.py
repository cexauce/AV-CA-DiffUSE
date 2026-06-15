import os
import sys
from tqdm import tqdm


import torch
sys.path.append(".")

sys.path.append(os.path.abspath("./sgmse/util/av_hubert/av_hubert"))
######
"""from sgmse.backbones.shared import BackboneRegistry
#from sgmse.data_module_av import SpecsDataModule
from sgmse.data_module_icp52 import SpecsDataModule
from sgmse.sdes import SDERegistry"""
from model import ScoreModel

ckpt_path = "./logs/av_diffse_late_fusion_avhubert_icp52_6M_infoNCE_trainable_audio_enc_warmup_100_alpha0_10000/last.ckpt"
device="cuda"
def similarity(subset):
    sim_matrix, files = model.evaluate_similarity(subset)
    print(len(files))
    with open("results/similarity matrix/similarity_matrix_filenames_"+ subset + ".txt", "w") as outfile:
        outfile.write("\n".join(files))
    print("similarity, ", sim_matrix.shape)
    torch.save(sim_matrix, "results/similarity matrix/similarity_matrix_"+ subset + ".pt")

def latent_embeddings(model, subset):
    audio, visual, files = model.compute_latent_embeddings(subset)
    with open("results/latent_embeddings/latent_embeddings_filenames_"+ subset + ".txt", "w") as outfile:
        outfile.write("\n".join(files))
    print("latent audio embeddings, ", audio.shape)
    print("latent visual embeddings, ", visual.shape)
    torch.save(audio, "results/latent_embeddings/latent_audio_embeddings_"+ subset + ".pt")
    torch.save(visual, "results/latent_embeddings/latent_visual_embeddings_"+ subset + ".pt")
    return audio, visual



if __name__ == "__main__":
    

    print('Loading pretrained model : ', ckpt_path)
    model = ScoreModel.load_from_checkpoint(ckpt_path).to(device)
    
    print("loaded to ", device)
    model.eval(no_ema=False)
    model.data_module.setup(stage='fit')
    model.data_module.transform_type = transform_type
    print("Audio encoder" , model.dnn.audio_processor)
    print("Visual encoder" , model.dnn.feature_extractor)

    # reinforce the fact of keeping the encoders fixed

    for param in model.parameters():
        param.requires_grad = False


    # compute metrics and similarity on train and validation set
    print("cross modal similarity matrix")
    similarity('train')
    similarity('val')
