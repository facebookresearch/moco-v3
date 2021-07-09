# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg

__all__ = [
    'vit_small', 
    'vit_base',
    'vit_large',
    'vit_huge',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, use_bn=False, no_cls_token=False, **kwargs):
        super().__init__(**kwargs)
        self.no_cls_token = no_cls_token

        # Use 2D sin-cos position embedding
        del self.pos_embed
        self.build_2d_sincos_position_embedding()

        if use_bn:
            self.replace_lns_with_bns()

        if no_cls_token:
            del self.cls_token
            self.num_tokens -= 1

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Hidden dimension must be divisible by 4 for 2D sin-cos position embedding.'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        if not self.no_cls_token:
            pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
            self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        else:
            self.pos_embed = nn.Parameter(pos_emb)
        self.pos_embed.requires_grad = False

    def replace_lns_with_bns(self):
        # replace LNs with BNs in the MLP blocks
        for blk in self.blocks:
            del blk.norm2
            blk.norm2 = nn.BatchNorm1d(self.embed_dim, eps=1e-6)

        # replace last LN with BN
        del self.norm
        self.norm = nn.BatchNorm1d(self.embed_dim, eps=1e-6)

    def forward_features(self, x):
        x = self.patch_embed(x)

        x_list = []
        if not self.no_cls_token:
            x_list.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.dist_token is not None:
            x_list.append(self.dist_token.expand(x.shape[0], -1, -1))
        x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        if self.no_cls_token:
            x_feat = x[:, self.num_tokens:].mean(dim=1) # take the mean over all tokens
        else:
            x_feat = x[:, 0]

        if self.dist_token is None:
            return self.pre_logits(x_feat)
        else:
            return x_feat, x[:, self.num_tokens-1]



def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_large(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def vit_huge(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model