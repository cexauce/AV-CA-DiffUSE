import os
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import cv2
import random
import argparse


def custom_collate_fn(batch):
    X_batch = [item[0] for item in batch]
    v_feats_batch = [item[1] for item in batch]

    # Pad X_batch to the maximum length
    X_lengths = [x.shape[-1] for x in X_batch]
    max_X_length = max(X_lengths)
    X_padded = torch.stack([F.pad(x, (0, max_X_length - x.shape[-1])) for x in X_batch])

    # Pad v_feats_batch to the maximum length
    v_feats_lengths = [v.shape[1] for v in v_feats_batch]
    max_v_feats_length = max(v_feats_lengths)
    v_feats_padded = torch.stack([
        F.pad(v, (0, 0, 0, max_v_feats_length - v.shape[1])) for v in v_feats_batch
    ])

    return X_padded, v_feats_padded
# DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, num_workers=4)

def my_resample(video, target_num):
    """
    Resample the video to have a target number of frames by selecting frames at calculated intervals.

    Parameters:
    - video: torch.Tensor of shape (N, W, H), where N is the number of frames.
    - target_num: int, the desired number of frames in the resampled video.

    Returns:
    - res: torch.Tensor of shape (target_num, W, H), the resampled video.
    """
    N, W, H = video.shape
    ratio = N / target_num
    idx_lst = torch.arange(target_num, dtype=torch.float32, device=video.device)
    idx_lst *= ratio
    idx_lst_int = idx_lst.long().clamp(max=N - 1)  # Ensure indices are within bounds
    res = video[idx_lst_int]
    return res

import torch

def my_pad(video, target_num):
    """
    Pad or truncate the video to have a target number of frames.

    Parameters:
    - video: torch.Tensor of shape (N, W, H), where N is the number of frames.
    - target_num: int, the desired number of frames in the output video.

    Returns:
    - res: torch.Tensor of shape (target_num, W, H), the padded or truncated video.
    """
    N, W, H = video.shape
    if N > target_num:
        # Truncate the video if the current number of frames is greater than the target
        res = video[:target_num]
    elif N < target_num:
        # Pad the video with zeros if the current number of frames is less than the target
        pad_size = target_num - N
        pad = torch.zeros((pad_size, W, H), dtype=video.dtype, device=video.device)
        res = torch.cat((video, pad), dim=0)
    else:
        # Return the original video if it already has the target number of frames
        res = video

    return res


def getTIMITclean(subset, data_dir="/group_storage/corpus/audio_visual/TCD-TIMIT/", t='_data_NTCD'):
    if subset == 'valid':
        subset = 'val'
    t1 = subset + t
    if subset == 'test':
        t1 = os.path.join(t1, 'clean') 
    clean_files = sorted([os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, t1)) for name in files if name.endswith('.wav')])  
    return clean_files

def getLRS3clean(subset, data_dir="/group_storage/corpus/audio_visual/LRS3_audios/"):
    if subset in ['train', 'valid']:
        subset = 'trainval'
    clean_files = sorted([os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, subset)) for name in files if name.endswith('.wav')])  
    
    print("{} files".format(len(clean_files)))
    return clean_files



def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            audio_only, video_feature_type = "raw_image", format='default', normalize="clean", spec_transform=None,
            stft_kwargs=None, return_time=False, spectogram_learning=False, 
            **ignored_kwargs):
        
        self.return_time = return_time
        self.subset = subset
        self.audio_only = audio_only
        self.video_feature_type = video_feature_type

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))

        elif format == "tcd-timit": 
            data_dir="/group_storage/corpus/audio_visual/TCD-TIMIT/" 
            t='_data_NTCD'
            self.clean_files = getTIMITclean(subset, data_dir, t)
            self.fps = 25
            self.num_vframes = int(self.fps * 2.04)
        
        elif format =="lrs3":

     
            #video_data_dir="/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/avhubert/"
            audio_dir="/group_storage/corpus/audio_visual/LRS3_audios/"
            print("format : ", format)

            
            self.clean_files = getLRS3clean(subset, audio_dir)
            #print(self.clean_files)
            self.fps = 25
            self.num_vframes = int(self.fps * 2.04)



        elif format == "wsj0":
            data_dir = "/group_storage/corpus/speech_recognition/wsj0_wav"
            dic = {
                "train": "**/si_tr_s/**/*.wav",
                "valid": "**/si_dt_05/**/*.wav",
                "test": "**/si_et_05/**/*.wav",
            }
            self.clean_files = sorted(glob(data_dir + dic[subset], recursive=True))
                    
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")
          
        if format =="tcd-timit":
            
            ##use the mouths cropped with index (48,68) and 88*88 roi,            
            if self.video_feature_type in ["avhubert", "resnet", "raw_image",]:
                self.video_size= 88
                self.video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"                
                            
            if self.video_feature_type == "flow_avse":
                self.video_size= 112
                self.video_path  = "/group_storage/corpus/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4" 

            if self.video_feature_type in ["resnet_pre"]:
                if subset == "valid":
                    self.video_path  = "/group_storage/corpus/audio_visual/TCD-TIMIT/val_data_NTCD/{speaker_id}/{filename}RawVF.npy"
                else:
                    self.video_path  = "/group_storage/corpus/audio_visual/TCD-TIMIT/{subset}_data_NTCD/{speaker_id}/{filename}RawVF.npy"
                self.video_size= 512

            if self.video_feature_type in ["avhubert_pre"]:
                if subset == "valid":
                    self.video_path  = "/group_storage/corpus/audio_visual/TCD-TIMIT/val_data_NTCD/{speaker_id}/{filename}_avhubert_fps30.npy"
                else:
                    self.video_path  = "/group_storage/corpus/audio_visual/TCD-TIMIT/{subset}_data_NTCD/{speaker_id}/{filename}_avhubert_fps30.npy"
                self.video_size= 768                

            if self.video_feature_type in ["cogenav"]:
                if subset == "valid":
                    self.video_path  = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/cogenav2/embeddings/{subset}_data_NTCD/{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                else:
                    self.video_path  = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/cogenav2/embeddings/{subset}_data_NTCD/{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                self.video_size= 768  
        elif format == 'lrs3':
            

            if self.video_feature_type in ["avhubert_pre", "avhubert"]:
                if subset in ["test", "valid"]:
                    self.video_path  = "/srv/storage/talc3@storage4.nancy/multispeech/calcul/users/cmboungo/avhubert_pre/LRS3/test_data/{speaker_id}/{filename}_avhubert.npy"
                else:
                    self.video_path = "/srv/storage/talc3@storage4.nancy/multispeech/calcul/users/cmboungo/avhubert_pre/LRS3/{subset}val_data/{speaker_id}/{filename}_avhubert.npy"        
            
            elif self.video_feature_type in ["cogenav"]:
                if subset  in ["test", "valid"]:
                    self.video_path  = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/cogenav_pre/LRS3/test_data/{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                else :
                    self.video_path = "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/calcul/users/cmboungo/cogenav_pre/LRS3/trainval_data/{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
            self.video_size= 768 
 
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.spectogram_learning = spectogram_learning
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * 2.04) #chunk_size

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        self.format = format 


    def videocap(self, path, start_frame):
        # Calculate the video start frame based on the audio start frame
        vid_start = int(start_frame // self.sample_rate * self.fps)
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            frames = []
            for i in range(vid_start + self.num_vframes):
                ret, img = cap.read()
                if i < vid_start:
                    continue

                if ret:
                    # Convert image to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # # Resize image
                    img = cv2.resize(img, (64, 64))
                    # Convert to PyTorch tensor and normalize
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    frames.append(img_tensor)
                else:
                    if i - vid_start < 30:
                        return None

            frame_tensor = torch.stack(frames)
            return frame_tensor
        else:
            # Handle video not opening correctly
            return None
        
    def vfeatscap(self, path, start_frame, thr_num=5):
        # Calculate the video start frame based on the audio start frame
        vid_start = int(start_frame // self.sample_rate * self.fps)
        vfeats = np.load(path)  # Shape: [feature_dim, total_frames]
        #print("path : ", path)
        #print("vfeats :  ", vfeats.shape)

        if self.video_feature_type in ["avhubert_pre"] and self.format!="lrs3":
            vfeats = vfeats.T

        total_frames = vfeats.shape[-1]
        num_vframes = self.num_vframes  # Desired number of frames (e.g., 51)

        vid_end = vid_start + num_vframes

        # Check if vid_start is within the bounds of the video
        if vid_start >= total_frames:
            # Not enough frames; return None to indicate the sample should be skipped
            return None

        # Slice vfeats from vid_start to vid_end
        vfeats_slice = vfeats[..., vid_start:vid_end]  # Shape: [feature_dim, num_available_frames]

        # Number of frames obtained
        num_available_frames = vfeats_slice.shape[-1]

        if num_available_frames < num_vframes:
            # Calculate the shortage
            shortage = num_vframes - num_available_frames

            if shortage <= thr_num:
                # Not enough frames but within acceptable threshold; pad by repeating the last frame
                last_frame = vfeats_slice[..., -1:]
                padding = np.repeat(last_frame, shortage, axis=-1)
                vfeats_padded = np.concatenate((vfeats_slice, padding), axis=-1)
            else:
                # Shortage exceeds threshold; return None to indicate the sample should be skipped
                return None
        else:
            # Enough frames; no padding needed
            vfeats_padded = vfeats_slice

        # Convert to torch tensor
        vfeats_tensor = torch.from_numpy(vfeats_padded)
        return vfeats_tensor


    def load_audio(self, file_path):
        try:
            x, _ = load(file_path) # mono as default
        except:
            return None, 0
        
        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start_frame = int(np.random.uniform(0, int(0.9 * (current_len - target_len) )))
            else:
                start_frame = int((current_len - target_len) / 2)
            x = x[..., start_frame : start_frame + target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            start_frame = 0
            x = F.pad(x, (0, pad), mode="constant") # pad only the end
        
        return x, start_frame
    
    def __getitem__(self, index):
        if self.format=='lr3' : 
            max_retries = 30
        else:
            max_retries = 10
        retries = 0

        while retries < max_retries:
            # Load audio data
            x, start_frame = self.load_audio(self.clean_files[index])
            if x is None:
                # Pick another random sample
                index = random.randint(0, len(self.clean_files) - 1)
                retries += 1
                continue

            # Load visual data
            speaker_id_i = self.clean_files[index].split("/")[-2]
            filename_i = self.clean_files[index].split("/")[-1].replace(".wav", "")
            video_path_i = self.video_path.format(
                subset=self.subset, speaker_id=speaker_id_i, filename=filename_i
            )

            v_feats = self.vfeatscap(video_path_i, start_frame)
            if v_feats is None:
                # Pick another random sample
                index = random.randint(0, len(self.clean_files) - 1)
                retries += 1
                continue

            # Permute and reshape v_feats
            if self.video_feature_type in ["resnet_pre", "avhubert_pre", "cogenav"]:
                v_feats = v_feats.permute(1, 0).contiguous().unsqueeze(0)

            # Normalize and process spectrogram
            if self.normalize == "clean":
                normfac = x.abs().max()
            elif self.normalize == "not":
                normfac = 1.0
            x = x / normfac
            X = torch.stft(x, **self.stft_kwargs)
            X = self.spec_transform(X)

            # Ensure tensors are contiguous and on CPU
            X = X.contiguous().cpu()
            v_feats = v_feats.contiguous().cpu()

            # Return the data
            #print(v_feats.shape)
            return X, v_feats

        # If max retries exceeded, raise an exception or handle accordingly
        raise Exception("Max retries exceeded in __getitem__")


    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, default="dummy", help="The base directory of the dataset. If `default` format then should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("default","tcd-timit", "lrs3", "dns", "wsj0"), default="tcd-timit", required=False, help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "not"), default="clean", help="Normalize the input waveforms by the clean signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none", "normalise"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--spectogram_learning", action='store_true', help="Train model on spectograms and use mixture signal for phase approximations")
        return parser

    def __init__(
        self, base_dir, audio_only, format='wsj0', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='clean', transform_type="exponent", 
        spectogram_learning=False, **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.spectogram_learning = spectogram_learning
        self.kwargs = kwargs
        self.audio_only = audio_only

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, 
                spectogram_learning=self.spectogram_learning, audio_only = self.audio_only, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, 
                spectogram_learning=self.spectogram_learning, audio_only = self.audio_only, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, return_time=True, 
                spectogram_learning=self.spectogram_learning, audio_only = self.audio_only, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "normalise":
            spec = spec / spec.abs().max()
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "normalise":
            spec = spec
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
    

def test_data_module(data_module_class, base_dir, format='tcd-timit', batch_size=4, num_workers=0, audio_only=False):
    parser = argparse.ArgumentParser()
    data_module_class.add_argparse_args(parser)
    args = parser.parse_args([])
    
    args.base_dir = base_dir
    args.format = format
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.audio_only = audio_only

    data_module = data_module_class(**vars(args))
    data_module.setup(stage='fit')

    val_loader = data_module.val_dataloader()
    print("Testing train dataloader...", len(val_loader))
    for i, (audio, visual) in enumerate(val_loader):
        print(f"Batch {i+1}: Audio shape: {audio.shape}, Visual shape: {visual.shape if visual is not None else 'N/A'}")
        if i >= 2:  # Limiting to 3 batches for quick testing
            break

if __name__ == "__main__":
    # Define the STFT parameters
    stft_kwargs = {
        'n_fft': 510,
        'hop_length': 128,
        'center': True,
        'window': get_window('hann', 510),
        'return_complex': True
    }

    # Now, create an instance of Specs with the correct stft_kwargs
    dataset = Specs(
        data_dir='/path/to/data',
        subset='train',
        dummy=False,
        shuffle_spec=False,
        num_frames=256,
        audio_only=False,
        video_feature_type='resnet_pre',
        format = "tcd-timit",
        normalize="clean",
        spec_transform=None,
        stft_kwargs=stft_kwargs,  # pass the correct stft_kwargs
        return_time=False,
        spectogram_learning=False
    )

    # Test __getitem__
    audio, visual = dataset.__getitem__(0)  # Access the first item
    print(f"Audio shape: {audio.shape}")
    if visual is not None:
        print(f"Visual shape: {visual.shape}")
    else:
        print("Visual data is None")