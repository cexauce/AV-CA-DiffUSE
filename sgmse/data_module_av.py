import os
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import cv2
import random
import argparse


def custom_collate_fn(batch):
    """
    Optional custom collate function if you want to pad audio & visual sequences in a batch.
    Not used by default in the DataModule below (you can plug it in if needed).
    """
    X_batch = [item[0] for item in batch]
    v_feats_batch = [item[1] for item in batch]

    # Pad X_batch to the maximum length
    X_lengths = [x.shape[-1] for x in X_batch]
    max_X_length = max(X_lengths)
    X_padded = torch.stack([F.pad(x, (0, max_X_length - x.shape[-1])) for x in X_batch])

    # If there is no visual data (audio_only), v_feats_batch will be list of None
    if v_feats_batch[0] is None:
        v_feats_padded = None
    else:
        # Pad v_feats_batch to the maximum length along time dimension
        v_feats_lengths = [v.shape[1] for v in v_feats_batch]
        max_v_feats_length = max(v_feats_lengths)
        v_feats_padded = torch.stack([
            F.pad(v, (0, 0, 0, max_v_feats_length - v.shape[1])) for v in v_feats_batch
        ])

    return X_padded, v_feats_padded


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
        # Truncate
        res = video[:target_num]
    elif N < target_num:
        # Pad with zeros
        pad_size = target_num - N
        pad = torch.zeros((pad_size, W, H), dtype=video.dtype, device=video.device)
        res = torch.cat((video, pad), dim=0)
    else:
        res = video
    return res


def getTIMITclean(subset, data_dir="/group_storage/corpus/audio_visual/TCD-TIMIT/", t='_data_NTCD'):
    if subset == 'valid':
        subset = 'val'
    t1 = subset + t
    if subset == 'test':
        t1 = os.path.join(t1, 'clean')
    clean_files = sorted([
        os.path.join(root, name)
        for root, dirs, files in os.walk(os.path.join(data_dir, t1))
        for name in files if name.endswith('.wav')
    ])
    return clean_files


def getLRS3clean(subset, data_dir="/group_storage/corpus/audio_visual/LRS3_audios/"):
    if subset in ['train', 'valid']:
        subset = 'trainval'
    clean_files = sorted([
        os.path.join(root, name)
        for root, dirs, files in os.walk(os.path.join(data_dir, subset))
        for name in files if name.endswith('.wav')
    ])
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
    def __init__(
        self, data_dir, subset, dummy, shuffle_spec, num_frames,
        audio_only, video_feature_type="raw_image", format='default',
        normalize="clean", spec_transform=None,
        stft_kwargs=None, return_time=False, spectogram_learning=False,
        **ignored_kwargs
    ):
        self.return_time = return_time
        self.subset = subset
        self.audio_only = audio_only
        self.video_feature_type = video_feature_type

        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))

        elif format == "tcd-timit":
            data_dir = "/group_storage/corpus/audio_visual/TCD-TIMIT/"
            t = '_data_NTCD'
            self.clean_files = getTIMITclean(subset, data_dir, t)
            self.fps = 25
            self.num_vframes = int(self.fps * 2.04)

        elif format == "lrs3":
            audio_dir = "/group_storage/corpus/audio_visual/LRS3_audios/"
            print("format : ", format)
            self.clean_files = getLRS3clean(subset, audio_dir)
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
            raise NotImplementedError(f"Directory format {format} unknown!")

        # Video path templates
        if format == "tcd-timit":
            # use the mouths cropped with index (48,68) and 88*88 roi
            if self.video_feature_type in ["avhubert", "resnet", "raw_image"]:
                self.video_size = 88
                self.video_path = (
                    
                    "corpus/audio_visual/CROPPED_MOUTH_ldmark_48_68_size_88_88/"
                    "TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"
                )

            if self.video_feature_type == "flow_avse":
                self.video_size = 112
                self.video_path = (
                    
                    "corpus/audio_visual/CROPPED_MOUTH_ldmark_28_68_size_112_112/"
                    "TCD-TIMIT/{subset}/{speaker_id}/straightcam/{filename}_mouthcrop.mp4"
                )

            if self.video_feature_type in ["resnet_pre"]:
                if subset == "valid":
                    self.video_path = (
                        
                        "corpus/audio_visual/TCD-TIMIT/val_data_NTCD/{speaker_id}/{filename}RawVF.npy"
                    )
                else:
                    self.video_path = (
                        
                        "corpus/audio_visual/TCD-TIMIT/{subset}_data_NTCD/{speaker_id}/{filename}RawVF.npy"
                    )
                self.video_size = 512

            if self.video_feature_type in ["avhubert_pre"]:
                if subset == "valid":
                    self.video_path = (
                        
                        "corpus/audio_visual/TCD-TIMIT/val_data_NTCD/{speaker_id}/{filename}_avhubert.npy"
                    )
                else:
                    self.video_path = (
                        
                        "corpus/audio_visual/TCD-TIMIT/{subset}_data_NTCD/{speaker_id}/{filename}_avhubert.npy"
                    )
                self.video_size = 768

            #if self.video_feature_type in ["cogenav"]: #$
            if self.video_feature_type in ["cogenav_pre"]: #$
                if subset == "valid":
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/"
                        "calcul/users/cmboungo/cogenav2/embeddings/{subset}_data_NTCD/"
                        "{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                    )
                else:
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/"
                        "calcul/users/cmboungo/cogenav2/embeddings/{subset}_data_NTCD/"
                        "{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                    )
                self.video_size = 768

        elif format == 'lrs3':
            if self.video_feature_type in ["avhubert_pre", "avhubert"]:
                if subset in ["test", "valid"]:
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy/multispeech/"
                        "calcul/users/cmboungo/avhubert_pre/LRS3/test_data/"
                        "{speaker_id}/{filename}_avhubert.npy"
                    )
                else:
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy/multispeech/"
                        "calcul/users/cmboungo/avhubert_pre/LRS3/{subset}val_data/"
                        "{speaker_id}/{filename}_avhubert.npy"
                    )

            #elif self.video_feature_type in ["cogenav"]:  
            elif self.video_feature_type in ["cogenav_pre"]:   #$ 
                if subset in ["test", "valid"]:
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy.grid5000.fr/"
                        "multispeech/calcul/users/cmboungo/cogenav_pre/LRS3/test_data/"
                        "{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                    )
                else:
                    self.video_path = (
                        "/srv/storage/talc3@storage4.nancy.grid5000.fr/"
                        "multispeech/calcul/users/cmboungo/cogenav_pre/LRS3/trainval_data/"
                        "{speaker_id}/{filename}_cogenav_feats_before_transformer.npy"
                    )
            self.video_size = 768

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.spectogram_learning = spectogram_learning
        self.sample_rate = 16000

        # audio chunk length in samples (≈ 2.04 s)
        self.chunk_size = int(self.sample_rate * 2.04)

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) is True, "'center' must be True for current implementation"
        self.format = format

    def videocap(self, path, start_frame):
        """
        Load a raw video segment aligned to the audio start_frame.
        Uses time in seconds to compute frame index, clamps at the end, and pads if needed.
        Returns tensor of shape [T, H, W] with T = self.num_vframes.
        """
        # Convert audio sample index to time, then to video frame index
        t0 = start_frame / float(self.sample_rate)  # seconds
        vid_start = int(round(t0 * self.fps))

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        # Clamp vid_start to avoid going out of range if possible
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            if total_frames <= self.num_vframes:
                vid_start = 0
            else:
                if vid_start + self.num_vframes > total_frames:
                    vid_start = total_frames - self.num_vframes
                vid_start = max(vid_start, 0)

        # Seek to vid_start (if supported)
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_start)

        frames = []
        for _ in range(self.num_vframes):
            ret, img = cap.read()
            if not ret:
                break

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize
            img = cv2.resize(img, (64, 64))
            # To tensor [H, W] in [0, 1]
            img_tensor = torch.from_numpy(img).float() / 255.0
            frames.append(img_tensor)

        cap.release()

        if len(frames) == 0:
            return None

        # If too short, pad with last frame
        if len(frames) < self.num_vframes:
            last = frames[-1]
            for _ in range(self.num_vframes - len(frames)):
                frames.append(last.clone())

        frame_tensor = torch.stack(frames, dim=0)  # [T, H, W]
        return frame_tensor

    def vfeatscap(self, path, start_frame):
        """
        Load precomputed visual features segment aligned to the audio start_frame.
        Uses time in seconds to compute the visual start index, clamps at boundaries,
        and pads with the last frame if needed. Returns [feature_dim, num_vframes] tensor.
        """
        # Convert audio sample index → time → visual frame index
        t0 = start_frame / float(self.sample_rate)  # seconds
        vid_start = int(round(t0 * self.fps))

        try:
            vfeats = np.load(path)  # assume [feat_dim, T] or [T, feat_dim], see below
        except Exception:
            # Truly unreadable file → signal failure
            return None

        # For non-LRS3 avhubert_pre, transpose to make time the last dimension
        """if self.video_feature_type in ["avhubert_pre"] and self.format != "lrs3":
            vfeats = vfeats.T  # now [feat_dim, T]"""

        # We assume time is always the last axis
        feature_dim = vfeats.shape[-2]
        total_frames = vfeats.shape[-1]
        num_vframes = self.num_vframes

        # Clamp vid_start to valid range
        if total_frames <= num_vframes:
            vid_start = 0
        else:
            if vid_start + num_vframes > total_frames:
                vid_start = total_frames - num_vframes
            vid_start = max(vid_start, 0)

        vid_end = vid_start + num_vframes
        vfeats_slice = vfeats[..., vid_start:vid_end]  # [..., T_slice]
        num_available = vfeats_slice.shape[-1]

        # If shorter, pad with last frame
        if num_available < num_vframes:
            shortage = num_vframes - num_available
            last_frame = vfeats_slice[..., -1:]  # [..., 1]
            padding = np.repeat(last_frame, shortage, axis=-1)  # [..., shortage]
            vfeats_padded = np.concatenate((vfeats_slice, padding), axis=-1)
        else:
            vfeats_padded = vfeats_slice

        return torch.from_numpy(vfeats_padded)

    def load_audio(self, file_path):
        """
        Load an audio chunk of length self.chunk_size (~2.04 s).
        Choose the start time in seconds, then convert to samples.
        Returns (audio_tensor, start_frame_samples).
        """
        try:
            x, _ = load(file_path)  # mono as default
        except Exception:
            return None, 0

        # Desired audio length in samples for STFT with center=True
        target_len = self.chunk_size  # equivalent to (num_frames - 1) * hop_length
        current_len = x.size(-1)

        if current_len >= target_len:
            # duration in seconds
            duration = current_len / float(self.sample_rate)
            seg_dur = target_len / float(self.sample_rate)

            t0_max = max(0.0, duration - seg_dur)
            if self.shuffle_spec and t0_max > 0:
                # 0.9 factor to avoid going too close to the end
                t0 = float(np.random.uniform(0.0, 0.9 * t0_max))
            else:
                t0 = t0_max / 2.0

            start_frame = int(round(t0 * self.sample_rate))
            end_frame = start_frame + target_len
            if end_frame > current_len:
                # clamp if rounding pushed us over
                start_frame = current_len - target_len
                start_frame = max(start_frame, 0)
                end_frame = start_frame + target_len

            x = x[..., start_frame:end_frame]
        else:
            # pad audio if too short
            pad = target_len - current_len
            start_frame = 0
            x = F.pad(x, (0, pad), mode="constant")

        return x, start_frame

    def __getitem__(self, index):
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

            # Normalize audio
            if self.normalize == "clean":
                normfac = x.abs().max()
            elif self.normalize == "not":
                normfac = 1.0
            else:
                normfac = 1.0
            x = x / (normfac + 1e-8)

            # Compute STFT and transform
            X = torch.stft(x, **self.stft_kwargs)
            if self.spec_transform is not None:
                X = self.spec_transform(X)

            # Load visual data if needed
            if self.audio_only:
                v_feats = None
            else:
                speaker_id_i = self.clean_files[index].split("/")[-2]
                filename_i = self.clean_files[index].split("/")[-1].replace(".wav", "")

                # Path for visual features or raw video
                video_path_i = self.video_path.format(
                    subset=self.subset, speaker_id=speaker_id_i, filename=filename_i
                )

                if self.video_feature_type in ["resnet_pre", "avhubert_pre", "cogenav_pre"]:#$ 
                    v_feats = self.vfeatscap(video_path_i, start_frame)
                else:
                    # raw video case
                    v_frames = self.videocap(video_path_i, start_frame)
                    if v_frames is None:
                        v_feats = None
                    else:
                        # you could plug a CNN / encoder later;
                        # for now treat raw frames as features [T, H, W]
                        v_feats = v_frames.unsqueeze(0)  # e.g. [1, T, H, W] if needed

                if v_feats is None:
                    # Try another sample
                    index = random.randint(0, len(self.clean_files) - 1)
                    retries += 1
                    continue

                # Permute to [1, T, F] for precomputed embeddings
                if self.video_feature_type in ["resnet_pre", "avhubert_pre", "cogenav_pre"]:#$ 
                    # v_feats: [feat_dim, T] -> [1, T, feat_dim]
                    v_feats = v_feats.permute(1, 0).contiguous().unsqueeze(0)


            # Ensure CPU and contiguous
            X = X.contiguous().cpu()
            if not self.audio_only and v_feats is not None:
                v_feats = v_feats.contiguous().cpu()

            return X, (v_feats if not self.audio_only else None)

        # If max retries exceeded, fail loudly
        raise Exception("Max retries exceeded in __getitem__")

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files) / 200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--base_dir", type=str, default="dummy",
            help="Base directory of the dataset."
        )
        parser.add_argument(
            "--format", type=str, choices=("default", "tcd-timit", "lrs3", "dns", "wsj0"),
            default="tcd-timit", required=False,
            help="Read file paths according to file naming format."
        )
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--n_fft", type=int, default=510)
        parser.add_argument("--hop_length", type=int, default=128)
        parser.add_argument("--num_frames", type=int, default=256)
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--dummy", action="store_true")
        parser.add_argument("--spec_factor", type=float, default=0.15)
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5)
        parser.add_argument("--normalize", type=str, choices=("clean", "not"), default="clean")
        parser.add_argument(
            "--transform_type", type=str,
            choices=("exponent", "log", "none", "normalise"), default="exponent"
        )
        parser.add_argument("--spectogram_learning", action='store_true')
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

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the window per device.
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

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "normalise":
            spec = spec / spec.abs().max()
        elif self.transform_type == "none":
            pass
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "normalise":
            pass
        elif self.transform_type == "none":
            pass
        return spec

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs,
            num_frames=self.num_frames,
            spec_transform=self.spec_fwd,
            **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(
                data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize,
                spectogram_learning=self.spectogram_learning,
                audio_only=self.audio_only, **specs_kwargs
            )
            self.valid_set = Specs(
                data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize,
                spectogram_learning=self.spectogram_learning,
                audio_only=self.audio_only, **specs_kwargs
            )
        if stage == 'test' or stage is None:
            self.test_set = Specs(
                data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, return_time=True,
                spectogram_learning=self.spectogram_learning,
                audio_only=self.audio_only, **specs_kwargs
            )

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
    print("Testing val dataloader...", len(val_loader))
    for i, (audio, visual) in enumerate(val_loader):
        print(f"Batch {i+1}: Audio shape: {audio.shape}, Visual shape: {visual.shape if visual is not None else 'None'}")
        if i >= 2:
            break


if __name__ == "__main__":
    # Example quick test
    # test_data_module(SpecsDataModule, base_dir="/path/to/LRS3_audios", format="lrs3", batch_size=2, num_workers=0, audio_only=False)
    pass
