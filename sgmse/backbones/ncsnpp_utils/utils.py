# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch

import numpy as np
from ...sdes import OUVESDE, OUVPSDE
from typing import Optional
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import librosa

class MultiModalAttention(nn.Module):
    def __init__(self, x_feature_dim, v_feature_dim, num_heads=1):
        super(MultiModalAttention, self).__init__()
        self.num_heads = num_heads
        self.x_feature_dim = x_feature_dim
        self.v_feature_dim = v_feature_dim
        self.head_dim = x_feature_dim // num_heads

        # Ensure the feature dimensions are compatible with the number of heads
        assert self.head_dim * num_heads == self.x_feature_dim, "x_feature_dim must be divisible by num_heads"

        # Key, value, and query transformation matrices
        self.W_q = nn.Linear(x_feature_dim, x_feature_dim, bias=False)
        self.W_k = nn.Linear(v_feature_dim, x_feature_dim, bias=False)
        self.W_v = nn.Linear(v_feature_dim, x_feature_dim, bias=False)
        # self.out_layer = nn.Linear(x_feature_dim, x_feature_dim)

    def forward(self, x, v):
        print("multi-modal attention forward : v ", v.shape)
        batch_size, _, temporal_dim_x, _ = x.shape
        
        _, _, temporal_dim_v, _ = v.shape
    
        # Transform v to get keys and values
        v = v.view(batch_size, temporal_dim_v, self.v_feature_dim)
        queries = self.W_q(x) # shape: (batch_size, temporal_dim_x, x_feature_dim)
        keys = self.W_k(v)   # shape: (batch_size, temporal_dim_v, x_feature_dim)
        values = self.W_v(v) # shape: (batch_size, temporal_dim_v, x_feature_dim)
        

        # Split keys, values, and queries for multi-head attention
        queries = queries.view(batch_size, temporal_dim_x, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, temporal_dim_v, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, temporal_dim_v, self.num_heads, self.head_dim).transpose(1, 2)
        

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
        attention = F.softmax(attention_scores, dim=-1)
        # np.save("attention_map.npy", attention.squeeze().detach().cpu().numpy())

        # Compute context vectors
        context = torch.matmul(attention, values).transpose(1, 2)
        context = context.contiguous().view(batch_size, temporal_dim_x, self.x_feature_dim)

        return context # self.out_layer(context)

def mask_spectrogram(spectrogram, n_mask=50, l_mask=5, p_cond=0.2):
    """
    Apply masking to a given spectrogram with batch and channel dimensions using PyTorch, fully vectorized.

    Parameters:
    spectrogram (torch.Tensor): 4D tensor representing the spectrogram, shape (batch, channel, frequency, time)
    n_mask (int): Number of frames to mask
    l_mask (int): Minimum length of contiguous frames to mask
    p_cond (float): Probability of masking a given frame

    Returns:
    torch.Tensor: The masked spectrogram
    """
    masked_spectrogram = spectrogram.clone()
    batch_size, num_channels, num_frequencies, num_time_frames = spectrogram.shape

    num_masks = n_mask // l_mask
    start_frames = torch.randint(0, num_time_frames - l_mask + 1, (batch_size, num_masks))
    mask_probs = torch.rand(batch_size, num_masks)
    mask = mask_probs < p_cond

    # Using broadcasting to apply the mask to the spectrogram
    for b in range(batch_size):
        for i in range(num_masks):
            if mask[b, i]:
                start = start_frames[b, i]
                masked_spectrogram[b, :, :, start:start + l_mask] = 0

    return masked_spectrogram


_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(sigma_min, sigma_max, num_scales):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(sigma_max), np.log(sigma_min), num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config.model.name
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, OUVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, OUVESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def audio_mel(x, audio_processor, target_sr=16000):
  """
  Mirror utils.load_audio + padding:
  - compute Whisper log-Mel
  - pad to batch tensor [1, 80, T]
  """
  #print("x", x.shape)   # [B=2, T=32640]
  mel = audio_processor.feature_extractor(
      x.detach().cpu().numpy(), sampling_rate=target_sr
  ).input_features  # [B, F=80, T'] or [B, T', F= 80] handled internally
  # padding
  #print("mel", len(mel), mel[0].shape) # mel (F=80, T'=3000)
  mel_padded = audio_processor.feature_extractor.pad(
      [{'input_features': mel}],
      return_tensors="pt"
  )['input_features']  
  #print("mel padded ", mel_padded.shape)# [B=2, F=80, T'=3000]
  return mel_padded