# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import math
import torch
import torch.nn.functional as F

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed_original(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def _infer_hw_from_num_tokens(n):
    r = int(math.sqrt(n))
    if r * r == n:
        return (r, r)
    return (1, n)  # fallback; better to pass target_hw explicitly

def _interpolate_pos_embed_any(pos_embed_checkpoint, num_extra_tokens, target_hw):
    """
    pos_embed_checkpoint: [1, N, C]
    num_extra_tokens: e.g., 1 for CLS, or 2 if there's also a dist token
    target_hw: (H', W') for the patch grid
    """
    embed_dim = pos_embed_checkpoint.shape[-1]
    extra = pos_embed_checkpoint[:, :num_extra_tokens]                # [1, T, C]
    grid  = pos_embed_checkpoint[:, num_extra_tokens:]                # [1, H*W, C]

    # infer original H,W (checkpoint is usually square for ImageNet-224)
    old_n = grid.shape[1]
    old_h = int(math.sqrt(old_n))
    old_w = old_h if old_h * old_h == old_n else old_n

    grid = grid.reshape(1, old_h, old_w, embed_dim).permute(0, 3, 1, 2)  # [1,C,H,W]
    grid = F.interpolate(grid, size=target_hw, mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, target_hw[0]*target_hw[1], embed_dim)
    return torch.cat([extra, grid], dim=1)

def interpolate_pos_embed(model, checkpoint_model, target_hw=None):
    """
    Rectangular-safe interpolation for both encoder and decoder pos embeds.
    - If target_hw is None, tries model.patch_embed.grid_size.
    """
    # figure out target grid H',W'
    if target_hw is None and hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        target_hw = tuple(model.patch_embed.grid_size)  # (H', W')
    if target_hw is None:
        # last resort: infer from model.num_patches (ambiguous if not square!)
        num_patches = model.patch_embed.num_patches
        target_hw = _infer_hw_from_num_tokens(num_patches)

    # ---- encoder ----
    if 'pos_embed' in checkpoint_model:
        pe = checkpoint_model['pos_embed']
        num_patches     = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # typically 1
        if pe.shape[-2] != (num_extra_tokens + target_hw[0]*target_hw[1]):
            checkpoint_model['pos_embed'] = _interpolate_pos_embed_any(
                pe, num_extra_tokens=num_extra_tokens, target_hw=target_hw
            )

    # ---- decoder ----
    if 'decoder_pos_embed' in checkpoint_model and hasattr(model, 'decoder_pos_embed'):
        pe = checkpoint_model['decoder_pos_embed']
        # same num_extra_tokens logic for decoder (usually 1)
        dec_num_patches  = model.decoder_pos_embed.shape[-2] - 1  # assume 1 extra token
        num_extra_tokens = model.decoder_pos_embed.shape[-2] - dec_num_patches
        if pe.shape[-2] != (num_extra_tokens + target_hw[0]*target_hw[1]):
            checkpoint_model['decoder_pos_embed'] = _interpolate_pos_embed_any(
                pe, num_extra_tokens=num_extra_tokens, target_hw=target_hw
            )
