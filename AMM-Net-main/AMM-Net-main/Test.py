import os
import math
from typing import Optional

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from transformers import BertTokenizer, BertModel

# 统一使用外部 attr_clip.py 中已经修好的实现
from attr_clip import RobustClipAttributeEncoder, AADB_PROMPTS_11

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# Swin Transformer
# =========================================================
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
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

        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(
                x,
                (
                    0, self.patch_size[1] - W % self.patch_size[1],
                    0, self.patch_size[0] - H % self.patch_size[0],
                    0, 0
                )
            )

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

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
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
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

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
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
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)
        return x


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    model = SwinTransformer(
        in_chans=3,
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=num_classes,
        **kwargs
    )
    return model


# =========================================================
# Text Encoder
# =========================================================
class EncoderText(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(
            embedding_dim,
            2048,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.25
        )

        self.out1 = nn.Linear(2048 * 2, 64)
        self.out2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, text):
        embedded = self.bert(text)[0]

        outs, hidden = self.rnn(embedded)
        outs = (outs[:, :, :outs.size(2) // 2] + outs[:, :, outs.size(2) // 2:]) / 2
        o = torch.mean(outs, dim=1)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = F.relu(self.out1(hidden))
        output = self.out2(output)
        return o, outs


# =========================================================
# Attention + MIMN
# =========================================================
class Attention_M(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1,
                 score_function='scaled_dot_product', dropout=0):
        super().__init__()

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
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
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
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)

        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]

        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)

        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)

        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class MIMN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hops = 3

        self.attention_text = Attention_M(2048, score_function='mlp')
        self.attention_img = Attention_M(2048, score_function='mlp')
        self.attention_text2img = Attention_M(2048, score_function='mlp')
        self.attention_img2text = Attention_M(2048, score_function='mlp')

        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)

    def forward(self, image_feature, attr_feature, txt_feature):
        et_text = attr_feature
        et_img = attr_feature

        for _ in range(self.hops):
            it_al_text2text = self.attention_text(txt_feature, et_text).squeeze(dim=1)
            it_al_img2text = self.attention_img2text(txt_feature, et_img).squeeze(dim=1)
            it_al_text = (it_al_text2text + it_al_img2text) / 2

            it_al_img2img = self.attention_img(image_feature, et_img).squeeze(dim=1)
            it_al_text2img = self.attention_text2img(image_feature, et_text).squeeze(dim=1)
            it_al_img = (it_al_img2img + it_al_text2img) / 2

            et_text = self.fc1(it_al_text)
            et_img = self.fc2(it_al_img)

        et = torch.cat((et_text, et_img), dim=-1)
        et = et.sum(dim=1)
        return et


# =========================================================
# catNet
# =========================================================
class catNet(nn.Module):
    def __init__(self, bert, freeze_clip=True):
        super().__init__()
        self.fusion = MIMN()
        self.txt_enc = EncoderText(bert)

        # 使用外部已修复版本，避免和 Test.py 内部重复实现冲突
        self.attr_enc = RobustClipAttributeEncoder(
            prompts=AADB_PROMPTS_11,
            out_dim=2048,
            freeze_clip=freeze_clip
        )

        self.img_enc = swin_base_patch4_window7_224_in22k(num_classes=10)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6144, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(1024, 2048)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text, image_att):
        txt_result, word_feature = self.txt_enc(text)

        img_feature = self.img_enc(image)     # (B, L, 1024)
        img_feature = self.fc4(img_feature)   # (B, L, 2048)

        img_attr = self.attr_enc(image_att)   # (B, 11, 2048)

        out = self.fusion(img_feature, img_attr, word_feature)
        h = torch.cat((out, txt_result), dim=1)

        h = self.drop(h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        h = self.softmax(h)
        return h


# =========================================================
# Metrics / Loss
# =========================================================
def binary_accuracy(y_pred, input_label, bins=10):
    rate_scale = torch.tensor([float(i + 1) for i in range(bins)], device=y_pred.device)
    threshold = float(bins / 2)
    pred_score = torch.sum(y_pred * rate_scale, dim=-1)
    true_score = torch.sum(input_label * rate_scale, dim=-1)
    diff = (((pred_score - threshold) * (true_score - threshold)) >= 0)
    acc = torch.sum(diff.float()) / pred_score.numel()
    return acc


def emd_dis(x, y_true, dist_r=1):
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
    score_pred = np.dot(output, np.arange(1, bins + 1))
    score_label = np.dot(target, np.arange(1, bins + 1))
    diff = (((score_pred - float(bins / 2)) * (score_label - float(bins / 2))) >= 0)
    acc_cls = np.sum(diff) / len(score_pred) * 100
    return [score_pred, score_label, acc_cls, output, target]


class emd_loss(nn.Module):
    def __init__(self, dist_r=2, use_l1loss=False, l1loss_coef=0.0):
        super().__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0

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


class AverageMeter:
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


# =========================================================
# Demo tokenizer / transforms
# 仅供 demo 推理使用；训练请走 dataset_ava.py
# =========================================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 200


def txt_process(txt):
    def pad(x):
        if len(x) > MAX_LEN:
            x = x[:MAX_LEN]
        else:
            x = x + [0] * (MAX_LEN - len(x))
        return x

    sentences = '[CLS] ' + txt + ' [SEP]'
    tokenized_sents = tokenizer.tokenize(sentences)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sents)
    input_ids = pad(input_ids)
    input_ids = torch.tensor(input_ids)
    return input_ids


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

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


# =========================================================
# Demo sample loader (import-safe)
# =========================================================
def load_demo_sample():
    imdir1 = './TestSet/Test_img.jpg'
    imdir2 = './TestSet/Test_text.txt'
    label_dir = './TestSet/Test_label.txt'

    img = Image.open(imdir1).convert("RGB")
    img_tensor = transform_test(img).unsqueeze(0)
    img_att = transform_att(img).unsqueeze(0)

    txt = open(imdir2, 'rb').read()
    txt = txt.decode('ascii', 'ignore')
    txt = txt_process(txt).unsqueeze(0)

    with open(label_dir, 'r') as f:
        listt = f.read().strip('\n').split(' ')
    label = [float(x) for x in listt[:10]]
    label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

    return img_tensor, txt, img_att, label


def test(mymodel, img_tensor, txt, img_att, label, device):
    mymodel.eval()
    criterion_aes_val = emd_loss(dist_r=1)
    scores_hist, labels_hist = [], []

    img_tensor = img_tensor.to(device)
    txt = txt.to(device)
    img_att = img_att.to(device)
    label = label.to(device)

    with torch.no_grad():
        output = mymodel(img_tensor, txt, img_att)
        _ = criterion_aes_val(output, label).item()
        acc_aes = binary_accuracy(output, label, 10)

    labels_hist.append(label.cpu().detach().numpy())
    scores_hist.append(output.cpu().detach().numpy())

    metrics = cal_metrics(scores_hist, labels_hist, 10)

    print(' --> Test_sample:')
    print(f'     - predicted score {metrics[0][0]:.4f} | true score {metrics[1][0]:.4f}')
    print(f'     - classification accuracy {metrics[2]:.1f}')
    print(f'     - predicted distribution {metrics[3][0]} | true distribution {metrics[4][0]}')

    return metrics[0], acc_aes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    bert = BertModel.from_pretrained('bert-base-uncased')
    mymodel = catNet(bert).to(device)

    model_path = "AMM-Net.pt"
    try:
        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and "model" in ckpt:
            incompat = mymodel.load_state_dict(ckpt["model"], strict=False)
        else:
            incompat = mymodel.load_state_dict(ckpt, strict=False)
        print("成功加载权重。")
        print("Missing keys:", incompat.missing_keys)
        print("Unexpected keys:", incompat.unexpected_keys)
    except FileNotFoundError:
        print("未找到 AMM-Net.pt 权重文件，将使用随机初始化测试。")

    if not (
        os.path.exists('./TestSet/Test_img.jpg') and
        os.path.exists('./TestSet/Test_text.txt') and
        os.path.exists('./TestSet/Test_label.txt')
    ):
        print("Demo test files not found. Skipping demo test.")
        return

    img_tensor, txt, img_att, label = load_demo_sample()
    test(mymodel, img_tensor, txt, img_att, label, device)


if __name__ == '__main__':
    main()