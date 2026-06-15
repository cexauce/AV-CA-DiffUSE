import os

import numpy as np

import scipy.stats
from scipy.signal import butter, sosfilt

import torch

import torch.nn as nn
import torch.nn.functional as F

from pesq import pesq
from pystoi import stoi


def si_sdr_components(s_hat, s, n):
    """ """
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s) ** 2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n) ** 2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise

    return s_target, e_noise, e_art


def energy_ratios(s_hat, s, n):
    """ """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10 * np.log10(
        np.linalg.norm(s_target) ** 2 / np.linalg.norm(e_noise + e_art) ** 2
    )
    si_sir = 10 * np.log10(np.linalg.norm(s_target) ** 2 / np.linalg.norm(e_noise) ** 2)
    si_sar = 10 * np.log10(np.linalg.norm(s_target) ** 2 / np.linalg.norm(e_art) ** 2)

    return si_sdr, si_sir, si_sar


def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


class Method:
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {}

        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value

    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))


def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off / sr * 2
    sos = butter(order, factor, "hp", output="sos")
    filtered = sosfilt(sos, signal)
    return filtered


def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s) / np.linalg.norm(s) ** 2
    sdr = 10 * np.log10(
        np.linalg.norm(alpha * s) ** 2 / np.linalg.norm(alpha * s - s_hat) ** 2
    )
    return sdr


def snr_dB(s, n):
    s_power = 1 / len(s) * np.sum(s**2)
    n_power = 1 / len(n) * np.sum(n**2)
    snr_dB = 10 * np.log10(s_power / n_power)
    return snr_dB


def pad_spec(Y):
    T = Y.size(3)
    if T % 64 != 0:
        num_pad = 64 - T % 64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
    return pad2d(Y)


def ensure_dir(file_path):
    directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_metrics(x, y, x_hat_list, labels, sr=16000):
    _si_sdr_mix = si_sdr(x, y)
    _pesq_mix = pesq(sr, x, y, "nb")
    _estoi_mix = stoi(x, y, sr, extended=True)
    print(
        f"Mixture:  PESQ: {_pesq_mix:.2f}, ESTOI: {_estoi_mix:.2f}, SI-SDR: {_si_sdr_mix:.2f}"
    )
    for i, x_hat in enumerate(x_hat_list):
        _si_sdr = si_sdr(x, x_hat)
        _pesq = pesq(sr, x, x_hat, "nb")
        _estoi = stoi(x, x_hat, sr, extended=True)
        print(f"{labels[i]}: {_pesq:.2f}, ESTOI: {_estoi:.2f}, SI-SDR: {_si_sdr:.2f}")


def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def print_mean_std(data, decimal=2):
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    if decimal == 2:
        string = f"{mean:.2f} ± {std:.2f}"
    elif decimal == 1:
        string = f"{mean:.1f} ± {std:.1f}"
    return string

class SigLIP(nn.Module):


    def __init__(self, **kwargs):
        super().__init__()

        params = {}
    
        # -- t_prime
        t_prime = kwargs.get("t_prime", None)
        if t_prime is None:
            # => if param is None, it means it should be learnable
            
            init_t_prime = np.log(10.)  # to match SigLIP paper init. t' value
            self.t_prime = nn.Parameter(torch.ones([]) * init_t_prime)
        else:
            # => if not, ensure that param is a torch tensor and not learnable - i.e. buffer - constant
            if isinstance(t_prime, (float, int)):
                self.register_buffer("t_prime", torch.tensor(float(t_prime)))
            else:
                raise ValueError("Wrong type for t_prime")
        params["t_prime"] = self.t_prime
        # -- b
        b = kwargs.get("b", None)
        if b is None:
            # => if param is None, it means it should be learnable
            init_b = -10.  # to match SigLIP paper init. b value
            self.b = nn.Parameter(torch.ones([]) * init_b)
        else:
            if isinstance(b, (float, int)):
                self.register_buffer("b", torch.tensor(float(b)))
            else:
                raise ValueError("Wrong type for b")
        params["b"] = self.b

        self.params = params



    def forward(self, X: torch.Tensor, Y: torch.Tensor, scaling_factor: float, **fw_kwargs):
        kwargs = {"t": self.params["t_prime"].exp(), "b": self.params["b"]}
        print("t_prime : ", self.params["t_prime"].item(), " b : ", self.params["b"].item())
        loss_func = getattr(self,"sigmoid_loss")
        return scaling_factor * loss_func(X, Y, **kwargs, **fw_kwargs)

    @staticmethod
    def sigmoid_loss(
            x: torch.Tensor, y: torch.Tensor, t: float, b: float,

    ):
        """
        Sigmoid loss from SigLIP paper https://arxiv.org/abs/2303.15343 (see Algorithm 1 pseudo implementation)

                    - 1 / B sum_i,j log 1 / ( 1 + e^{label_ij (-t <xi,yj> + bias)} )

                                    with label_ij = 1 if i=j else -1

        Warning: in the following implementation, to respect SigLIP pseudocode, the above formula `bias`
        corresponds to `-b`.
        """

        # reminder: x and y must have same shape (B, D)
        assert x.shape == y.shape

        # -- L2 normalization
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        # -- loss computation
        print("t : ", t.item(), " b : ", b.item())
        
        logits = t * (x @ y.t()) + b  # shape=(B, B)  | theoretically : t * (x @ y.t()) - b (but seems to work better with "+")
        n = logits.size(0)  # =B
        labels = 2 * torch.eye(n, device=logits.device) - torch.ones(n, device=logits.device)  # -1 BxB matrix with diagonal of 1
        # N.B: F.logsigmoid is f(x) := log 1 / ( 1 + exp(-x) )
        # (according to https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html)
        log_sig = F.logsigmoid(labels * logits)  # (B, B) | "*"=piecewise multiplication to retrieve xi yj
        print("log_sig : ", log_sig)
        return  - torch.sum(log_sig) / n 
