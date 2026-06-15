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

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import math

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method="cat"):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == "cat":
            return torch.cat([h, y], dim=1)
        elif self.method == "sum":
            return h + y
        else:
            raise ValueError(f"Method {self.method} not recognized.")


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6
        )
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class Upsample(nn.Module):
    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    up=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            h = F.interpolate(x, (H * 2, W * 2), "nearest")
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            if not self.with_conv:
                h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    def __init__(
        self,
        in_ch=None,
        out_ch=None,
        with_conv=False,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        else:
            if with_conv:
                self.Conv2d_0 = up_or_down_sampling.Conv2d(
                    in_ch,
                    out_ch,
                    kernel=3,
                    down=True,
                    resample_kernel=fir_kernel,
                    use_bias=True,
                    kernel_init=default_init(),
                )
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        else:
            if not self.with_conv:
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                x = self.Conv2d_0(x)

        return x


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        conv_shortcut=False,
        dropout=0.1,
        skip_rescale=False,
        init_scale=0.0,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)


class ResnetBlockBigGANpp(nn.Module):
    def __init__(
        self,
        act,
        in_ch,
        out_ch=None,
        temb_dim=None,
        up=False,
        down=False,
        dropout=0.1,
        fir=False,
        fir_kernel=(1, 3, 3, 1),
        skip_rescale=True,
        init_scale=0.0,
    ):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
        )
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(
            num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6
        )
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))

        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
            else:
                h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
                x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)

        if self.in_ch != self.out_ch or self.up or self.down:
            x = self.Conv_2(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)

class AudioResNetEmbedder(nn.Module):
    def __init__(self, in_ch=2, emb_dim=256):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # output: [B, feat_dim]

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        #print("Audio reset input x shape :", x.shape) # [256, 256, 2]
        h = self.backbone(x)          # [B, feat_dim]
        z = self.proj(h)              # [B, emb_dim]
        return z 


class ResBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock2D, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out
class AudioFront(nn.Module):

    
    def __init__(self, in_ch=1, emb_dim=768):
        super().__init__()
    # two 2D convolution layers 
        self.conv1 = nn.Conv2d(in_ch, 128, 3, 2, bias=False) # nn.Conv2d(in_ch, 128, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(128, 256, 3, 2, bias=False)
        
    # followed by one ResBlock
        self.resblock2D = ResBlock2D(inplanes=256, planes = 256)


    # flatten linear
        self.emb_dim = emb_dim
        self.flat = nn.Flatten(start_dim=-2, end_dim=-1)

        self.lin = nn.Linear(in_features=256*63, out_features=emb_dim)
    def forward(self, x):
        # x : [B, 1, F=256, T= 256]
        x1 = self.conv1(x) 
        print("x1 ", x1.shape )# x1 : [B, C = 128, F=127, T=127]
        x2 = self.conv2(x1)
        print("x2 ", x2.shape)# x2 : [B, C =256, F=63, T=63]
        h = self.resblock2D(x2)
        print("h ", h.shape)  # h : [B, C =256, F=63, T=63]
        h = h.permute(0, 2, 1, 3)            # h : [B,  T=63, C= 256,F=63]
          # h_flat : [B,T = 63, 256*63]
        h_flat = self.flat(h)
        return self.lin(h_flat)

class vCAFE_maskNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv1d(in_channels=768,out_channels=16 , kernel_size=3, stride=1,padding=2, dilation = 2)
        #self.conv2 = nn.Conv1d(in_channels=16,out_channels=768 , kernel_size=3, stride=1,padding=2, dilation = 2)
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, padding=2,dilation=2)
        self.conv2 = nn.Conv1d(in_channels=768,out_channels=768,kernel_size=3,padding=2,dilation=2)
        self.relu = nn.ReLU()
    def forward(self, c):
        print("c", c.shape) # [B, T, D2=768]
        c = self.conv1(torch.squeeze(c).permute(0, 2, 1)) # [B, D2=768, T]
        print("c", c.shape)
        mask = torch.sigmoid(self.conv2(self.relu(c)))
        print("mask", mask.shape)  # [B, T, D2]
        return mask

# Copyright 2023 Korea Advanced Institute of Science and Technology (Joanna Hong, Minsu Kim)

class VCAFE_Masking(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.softmax = nn.Softmax(2)
        self.k = nn.Linear(512, out_dim)
        self.v = nn.Linear(512, out_dim)
        self.q = nn.Linear(512, out_dim)
        self.out_dim = out_dim

        self.sigmoid = nn.Sigmoid()
        self.mask = nn.Sequential(
            nn.Conv1d(out_dim, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.fusion = nn.Linear(1024, 512)

    def forward(self, aud, vid, v_len):
        #aud: B,T,512
        #vid: B,S,512
        q = self.q(aud)  # B,T,OD
        k = self.k(vid).transpose(1, 2).contiguous()   # B,OD,S

        att = torch.bmm(q, k) / math.sqrt(self.out_dim)    # B,T,S
        for i in range(att.size(0)):
            att[i, :, v_len[i]:] = float('-inf')
        att = self.softmax(att)  # B,T,S

        v = self.v(vid)  # B,S,OD
        value = torch.bmm(att, v)  # B,T,OD

        mask = self.mask(value.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        for i in range(aud.size(0)):
            mask[i, v_len[i]:, :] = float('-inf')
        mask = self.sigmoid(mask)
        enhance_aud = aud * mask
        enhanced_aud = enhance_aud + aud

        fusion = torch.cat([enhanced_aud, vid], 2)
        fusion = self.fusion(self.dropout(fusion))

        return fusion