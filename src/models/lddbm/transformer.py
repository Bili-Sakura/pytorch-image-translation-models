# Copyright (c) 2025 Copyright holder of the paper "Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge" submitted to NeurIPS 2025 for review.
# All rights reserved.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Patches               #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbedding(nn.Module):
    """Patchify image and create embeddings"""

    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> Patch embeddings (B, N, D)
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = rearrange(x, "b d h w -> b (h w) d")  # (B, N, D)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Create 2D sinusoidal positional embeddings.

    Args:
        embed_dim: embedding dimension
        grid_size: size of spatial grid (height and width)
        cls_token: whether to include class token
        extra_tokens: number of extra tokens to prepend

    Returns:
        pos_embed: positional embeddings of shape [grid_size*grid_size, embed_dim]
                   or [extra_tokens+grid_size*grid_size, embed_dim] if extra_tokens > 0
    """
    # Create coordinate grids
    y_coords = np.arange(grid_size, dtype=np.float32)
    x_coords = np.arange(grid_size, dtype=np.float32)

    # Create meshgrid with y (height) and x (width)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Flatten the grids
    yy_flat = yy.flatten()
    xx_flat = xx.flatten()

    # Generate embeddings for each dimension
    half_dim = embed_dim // 2
    emb_y = get_1d_sincos_pos_embed_from_grid(half_dim, yy_flat)
    emb_x = get_1d_sincos_pos_embed_from_grid(half_dim, xx_flat)

    # Concatenate embeddings
    pos_embed = np.concatenate([emb_y, emb_x], axis=1)

    # Add extra tokens if needed
    if cls_token and extra_tokens > 0:
        extra_embed = np.zeros([extra_tokens, embed_dim], dtype=np.float32)
        pos_embed = np.concatenate([extra_embed, pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim: output dimension (must be even)
        grid: tuple of (grid_h, grid_w) where each is a flattened array of positions

    Returns:
        emb: positional embeddings of shape (H*W, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # Split embedding dimension equally between height and width
    dim_per_axis = embed_dim // 2

    # Generate embeddings for height and width separately
    height_emb = get_1d_sincos_pos_embed_from_grid(dim_per_axis, grid[0])
    width_emb = get_1d_sincos_pos_embed_from_grid(dim_per_axis, grid[1])

    # Combine height and width embeddings
    combined_emb = np.concatenate([height_emb, width_emb], axis=1)

    return combined_emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        embed_dim: output dimension for each position (must be even)
        pos: array of positions to be encoded, shape (M,)

    Returns:
        emb: positional embeddings of shape (M, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # Flatten position array
    pos = pos.flatten()

    # Create frequency bands
    half_dim = embed_dim // 2
    freq_bands = np.arange(half_dim, dtype=np.float32)
    freq_bands = np.exp(-np.log(10000.0) * freq_bands / half_dim)

    # Compute angles: pos * freq for each position and frequency
    angles = pos[:, np.newaxis] * freq_bands[np.newaxis, :]

    # Apply sine and cosine
    sin_emb = np.sin(angles)
    cos_emb = np.cos(angles)

    # Interleave sine and cosine: [sin, cos, sin, cos, ...]
    emb = np.concatenate([sin_emb, cos_emb], axis=1)

    return emb


class TransformerEncoder(nn.Module):
    """Transformer Encoder with timestep embedding"""

    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        # x: (B, N, D), t_emb: (B, D)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, N, D)


class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.nheads = num_heads

    def _split_heads(self, x):
        B, N, E = x.shape

        return x.view(B, N, self.nheads, E // self.nheads).transpose(1, 2)

    def _recombine_heads(self, x):
        B, _, N, _ = x.shape

        return x.transpose(1, 2).flatten(2)

    def forward(self, x, x_cross=None):
        if x_cross is None:
            k = self._split_heads(self.k_proj(x))
            q = self._split_heads(self.q_proj(x))
            v = self._split_heads(self.v_proj(x))
        else:
            k = self._split_heads(self.k_proj(x_cross))
            q = self._split_heads(self.q_proj(x))
            v = self._split_heads(self.v_proj(x_cross))

        # attn_mask = self._get_attn_mask(q.shape[2], k.shape[2], k.device)
        attn_out = self._recombine_heads(F.scaled_dot_product_attention(q, k, v))

        return self.o_proj(attn_out)


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder with timestep embedding"""

    def __init__(self, embed_dim, num_heads, ca_mod):
        super().__init__()
        self.ca_mod = ca_mod
        if ca_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim, 9 * embed_dim, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim, bias=True)
            )

        # Self Attention
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.sa = Attention(embed_dim, num_heads)

        # Cross Attention projections
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.ca = Attention(embed_dim, num_heads)

        # MLP for FFN
        self.norm4 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = 4 * embed_dim
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, memory, t_emb):  # t_emb should be

        if self.ca_mod:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mca,
                scale_mca,
                gate_mca,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(t_emb).chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t_emb).chunk(6, dim=1)
            )

        # Compute Self-Attention
        x_attn = modulate(self.norm1(x), shift_msa, scale_msa)  # B, 2N, E
        attn_out = self.sa(x_attn)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Compute Cross-Attention
        if self.ca_mod:
            x_attn = modulate(self.norm2(x), shift_mca, scale_mca)
            mem_attn = modulate(self.norm3(memory), shift_mca, scale_mca)
            attn_out = self.ca(x_attn, mem_attn)
            x = x + gate_mca.unsqueeze(1) * attn_out
        else:
            attn_out = self.ca(self.norm2(x), self.norm3(memory))
            x = x + attn_out

        # Compute FF
        attn_out = modulate(self.norm4(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(attn_out)

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, ca_mod):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, ca_mod)
                for _ in range(depth)
            ]
        )

    def forward(self, x, memory, t_emb):
        for layer in self.layers:
            x = layer(x, memory, t_emb)
        return x  # (B, N, D)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AutoRegressiveTransformer(nn.Module):

    def __init__(
        self,
        in_channels=160,
        img_size=8,
        patch_size=1,
        embed_dim=256,
        num_heads=8,
        depth=6,
        ca_mod=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.out_channels = in_channels

        self.x_embedder = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.t_embedder = TimestepEmbedder(embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        mask_token = torch.randn(1, 1, embed_dim)
        self.masks = nn.Parameter(torch.cat([mask_token] * self.num_patches, dim=1))

        self.encoder = TransformerEncoder(embed_dim, num_heads, depth)
        self.decoder = TransformerDecoder(embed_dim, num_heads, depth, ca_mod)

        self.out_proj = FinalLayer(embed_dim, patch_size, in_channels)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, xt, t, xT):
        B, C, H, W = xt.shape
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), "Image must be divisible by patch size."

        # Patch embeddings
        xT_tokens = self.x_embedder(xT) + self.pos_embed  # (B, N, D)
        memory = self.encoder(xT_tokens)

        t_emb = self.t_embedder(t)

        # x0_tokens = self.x_embedder(aux_cond) + self.pos_embed  # (B, N, D)
        x0_tokens = torch.repeat_interleave(self.masks, B, dim=0) + self.pos_embed
        xt_tokens = self.x_embedder(xt) + self.pos_embed  # (B, N, D)
        x = torch.cat((xt_tokens, x0_tokens), dim=1)

        _, output = self.decoder(x, memory, t_emb).chunk(2, dim=1)
        output = self.out_proj(output, t_emb)
        return output.reshape(B, C, H, W)
