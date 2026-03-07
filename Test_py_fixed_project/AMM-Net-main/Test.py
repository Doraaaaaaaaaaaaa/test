import os
import time
import random
import numpy as np
from tqdm import tqdm
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext import data
from transformers import BertTokenizer, BertModel
import cv2
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
import torchvision.models as models
import copy
import math
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torchtext
from scipy import io as sio
from torchvision import transforms
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from PIL import ImageFile
import torch.utils.model_zoo as model_zoo
from torchvision import transforms as tfs
ImageFile.LOAD_TRUNCATED_IMAGES = True
from functools import partial
from typing import Optional
import clip


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc1_1 = nn.Linear(512 * block.expansion, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.fc3_1 = nn.Linear(64, 11)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward


def _resnet2(arch, block, layers, pretrained, progress, **kwargs):
    model2 = ResNet2(block, layers, **kwargs)
    return model2


def resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet2('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


#####################################################################################################################

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding

        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
 
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:

            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head1 = nn.Linear(1024, 64)
        self.head2 = nn.Linear(64, num_classes)
        self.drop1 = nn.Dropout(0.25)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        # x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        # x = torch.flatten(x, 1)
        # # x = self.head(x)
        return x


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


#############################################################################################################

class EncoderText(nn.Module):
    def __init__(self, bert):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim,
                          2048,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.25)

        # self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.out1 = nn.Linear(2048 * 2, 64)
        self.out2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, text):

        # text = [batch size, sent len]

        embedded = self.bert(text)[0]

        # embedded = [batch size, sent len, emb dim]

        outs, hidden = self.rnn(embedded)
        outs = (outs[:, :, :outs.size(2) // 2] + outs[:, :, outs.size(2) // 2:]) / 2
        o = torch.mean(outs, dim=1)

        # hidden = [n layers * n directions, batch size, emb dim]
        # _ = [batch size, sent len, hid dim * 2]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        # output = self.out(hidden)
        output = F.relu(self.out1(hidden))
        output = self.out2(output)

        # output = [batch size, out dim]

        return o, outs


class Attention_M(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention_M, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv)
        self.w_qx.data.uniform_(-stdv, stdv)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
       
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output


class MIMN(nn.Module):
    def locationed_memory(self, memory, memory_len):
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1-float(idx)/int(memory_len[i]))
        return memory

    def __init__(self):
        super(MIMN, self).__init__()
        self.hops = 3

        self.attention_text = Attention_M(2048, score_function='mlp')
        self.attention_img = Attention_M(2048, score_function='mlp')
        self.attention_text2img = Attention_M(2048, score_function='mlp')
        self.attention_img2text = Attention_M(2048, score_function='mlp')

        self.gru_cell_text = nn.GRUCell(2048, 2048)
        self.gru_cell_img = nn.GRUCell(2048, 2048)

        self.bn = nn.BatchNorm1d(2048, affine=False)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)

    def forward(self, image_feature, attr_feature, txt_feature):
        et_text = attr_feature      
        et_img = attr_feature

        for _ in range(self.hops):
            it_al_text2text = self.attention_text(txt_feature, et_text).squeeze(dim=1)
            it_al_img2text = self.attention_img2text(txt_feature, et_img).squeeze(dim=1)
            it_al_text = (it_al_text2text + it_al_img2text)/2

            it_al_img2img = self.attention_img(image_feature, et_img).squeeze(dim=1)
            it_al_text2img = self.attention_text2img(image_feature, et_text).squeeze(dim=1)
            it_al_img = (it_al_img2img + it_al_text2img)/2
       
            et_text = self.fc1(it_al_text)
            et_img = self.fc2(it_al_img)

        et = torch.cat((et_text, et_img), dim=-1)
        et = et.sum(dim=1)

        return et



AADB_PROMPTS = [
    "a photo with interesting content",
    "a photo with clear object emphasis",
    "a photo with good lighting",
    "a photo with good color harmony",
    "a photo with vivid color",
    "a photo with shallow depth of field",
    "a photo with motion blur",
    "a photo following rule of thirds",
    "a photo with balanced elements",
    "a photo with repetition patterns",
    "a photo with symmetry"
]

class RobustClipAttributeEncoder(nn.Module):
    def __init__(self, prompts, out_dim=2048, clip_name="ViT-B/16", freeze_clip=True, temperature=0.07, device=None):
        super().__init__()
        self.temperature = temperature
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.clip_model, _ = clip.load(clip_name, device=self.device)
        self.clip_model.eval()
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            t = self.clip_model.encode_text(tokens)
            t = F.normalize(t, dim=-1)
        self.register_buffer("prompt_emb", t)

        d_clip = t.shape[-1]
        self.proj_T = nn.Linear(d_clip, out_dim)
        self.proj_V = nn.Linear(d_clip, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, img_clip):
        img_clip = img_clip.to(self.prompt_emb.device)
        with torch.no_grad():
            v = self.clip_model.encode_image(img_clip)
            v = F.normalize(v, dim=-1)
            logits = (v @ self.prompt_emb.t()) / self.temperature
            W = torch.softmax(logits, dim=-1)

        T_proj = self.proj_T(self.prompt_emb)
        T_proj = T_proj.unsqueeze(0).expand(v.size(0), -1, -1)

        V_proj = self.proj_V(v)
        V_gated = V_proj.unsqueeze(1) * W.unsqueeze(-1)

        F_a = self.layer_norm(T_proj + V_gated)
        return F_a

class catNet(nn.Module):
    def __init__(self, bert):
        super(catNet, self).__init__()
        # defining layers in catNet
        self.fusion = MIMN()
        self.txt_enc = EncoderText(bert)
        self.attr_enc = RobustClipAttributeEncoder(AADB_PROMPTS, out_dim=2048)
        self.img_enc = swin_base_patch4_window7_224_in22k(num_classes=10)
       
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6144, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(1024, 2048)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text,image_att):
        txt_result, word_feature = self.txt_enc(text)
        img_attr = self.img_attr(image_att)
        img_feature = self.img_enc(image)
        img_feature = self.fc4(img_feature)

        img_attr = img_attr.view(img_attr.size(0), img_attr.size(1), -1)
        img_attr = self.fc3(img_attr)
        img_attr = img_attr.transpose(2, 1)

        out = self.fusion(img_feature, img_attr, word_feature)

        h = torch.cat((out, txt_result), 1)

        h = self.drop(h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        h = self.softmax(h)

        return h


def binary_accuracy(y_pred, input_label, bins=10):
    rate_scale = torch.tensor([float(i+1) for i in range(bins)], device=y_pred.device)
    threshold  = float(bins / 2)
    _pred = torch.sum(y_pred * rate_scale, dim=-1)
    _label = torch.sum(input_label * rate_scale, dim=-1)
    diff = (((_pred-threshold) * (_label-threshold)) >= 0)
    acc = torch.sum(diff.float()) / _pred.numel()
    return acc

def emd_dis(x, y_true, dist_r = 1):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    loss = torch.mean(samplewise_emd)
    return loss

def cal_metrics(output, target, bins=10):
    output = np.concatenate(output)
    target = np.concatenate(target)
    score_pred = np.dot(output, np.arange(1, bins+1))
    score_label = np.dot(target, np.arange(1, bins+1))
    # srcc, _ = spearmanr(scores_pred, score_label)
    # plcc, _ = pearsonr(scores_pred, score_label)
    # mse = ((scores_pred - score_label)**2).mean(axis=None)
    diff = (((score_pred-float(bins/2)) * (score_label-float(bins/2))) >= 0)
    acc_cls = np.sum(diff) / len(score_pred) * 100
    # output_tensor = torch.from_numpy(output).cuda()
    # target_tensor = torch.from_numpy(target).cuda()
    output_tensor = torch.from_numpy(output)
    target_tensor = torch.from_numpy(target)
    # with torch.no_grad():
    #     emd1 = emd_dis(output_tensor, target_tensor, dist_r = 1)
    #     emd2 = emd_dis(output_tensor, target_tensor, dist_r = 2)
        # return [mse, srcc, plcc, acc, emd1, emd2]
    return [score_pred, score_label, acc_cls, output, target]


class emd_loss(torch.nn.Module):
    """
    Earth Mover Distance loss
    """
    def __init__(self, dist_r=2,
        use_l1loss=False, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        # assert len(x_type) == 2
        # assert len(y_type) == 2

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))
        if y_true.size()[1] == 5:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.2
        else:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.1
        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        samplewise_emd = samplewise_emd.mul(coff)
        loss = torch.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], device=x.device)
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss_coef = 1.0 - torch.abs(y_true_mean - 0.5)
            l1 = (x_mean - y_true_mean).pow(2)
            l1loss = torch.mean(l1.mul(l1loss_coef))
            loss += l1loss
        return loss

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name}:{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN=200
def txt_process(txt):
    def pad(x):
        if len(x) > MAX_LEN:
            x = x[:MAX_LEN]
        else:
            x = x + [0] * (MAX_LEN - len(x))
        # return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
        return x

    sentencses = '[CLS] ' + txt + ' [SEP]'
    tokenized_sents = tokenizer.tokenize(sentencses)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sents)

    input_ids = pad(input_ids)
    input_ids = torch.tensor(input_ids)

    return input_ids




def load_demo_sample():
    imdir1 = './TestSet/Test_img.jpg'
    imdir2 = './TestSet/Test_text.txt'
    
    img = Image.open(imdir1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(448, 448)),
        transforms.ToTensor(),
        normalize
    ])
    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    transform_att = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        clip_normalize
    ])
    
    img_tensor = transform_test(img).unsqueeze(0)
    img_att = transform_att(img).unsqueeze(0)
    
    txt = open(imdir2,'rb').read()
    txt = txt.decode('ascii', 'ignore')
    txt = txt_process(txt).unsqueeze(0)
    
    f=open('./TestSet/Test_label.txt').read()
    listt = f.strip('\n').split(' ')
    label = [float(listt[0]),float(listt[1]),float(listt[2]),float(listt[3]),float(listt[4]),float(listt[5]),float(listt[6]),float(listt[7]),float(listt[8]),float(listt[9])]
    label = torch.tensor(label).unsqueeze(0)
    return img_tensor, txt, img_att, label

def test(mymodel,img_tensor,txt,img_att,label,device):
    accs_aes = AverageMeter('Acc', ':.2f')
    mymodel.eval()
    # criterion_aes_val = nn.CrossEntropyLoss()
    criterion_aes_val = emd_loss(dist_r=1)
    test_loss = 0
    test_image_correct = 0
    scores_hist, labels_hist = [], []

    img_tensor, txt, img_att, label = img_tensor.to(device), txt.to(device), img_att.to(device), label.to(device)
    output = mymodel(img_tensor, txt, img_att)
    # target = target.squeeze()
    loss = criterion_aes_val(output, label).item()
    # loss = F.l1_loss(output, target)
    # image_preds = torch.argmax(output, 1)
    # all_test_loss.append(loss)
    acc_aes = binary_accuracy(output, label, 10)
    # accs_aes.update(acc_aes.cpu().data.numpy())

    labels_hist.append(label.cpu().detach().numpy())
    scores_hist.append(output.cpu().detach().numpy())

    metrics = cal_metrics(scores_hist, labels_hist, 10)

    print(f' --> Test_sample:')
    print(f'     - predicted score {metrics[0][0]:.4f} | true score {metrics[1][0]:.4f}')
    print(f'     - classification accuracy {metrics[2]:.1f} ')
    print(f'     - predicted distribution {metrics[3][0]} | true distribution {metrics[4][0]}')

    return metrics[0], acc_aes


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert = BertModel.from_pretrained('bert-base-uncased')

    mymodel = catNet(bert)

    model_path = "AMM-Net.pt"
    mymodel_dict = torch.load(model_path, map_location='cpu')
    mymodel.load_state_dict(mymodel_dict)

    mymodel = mymodel.to(device)

    img_tensor, txt, img_att, label = load_demo_sample()
    score, acc_aes = test(mymodel, img_tensor, txt, img_att, label, device)


if __name__ == '__main__':
    main()
