from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


IMF_PRESET_CONFIGS: Dict[str, Dict[str, object]] = {
    "iMF-B/2": {
        "sample_size": 32,
        "patch_size": 2,
        "hidden_size": 768,
        "depth": 12,
        "num_attention_heads": 12,
        "aux_head_depth": 8,
    },
    "iMF-M/2": {
        "sample_size": 32,
        "patch_size": 2,
        "hidden_size": 768,
        "depth": 24,
        "num_attention_heads": 12,
        "aux_head_depth": 8,
    },
    "iMF-L/2": {
        "sample_size": 32,
        "patch_size": 2,
        "hidden_size": 1024,
        "depth": 32,
        "num_attention_heads": 16,
        "aux_head_depth": 8,
    },
    "iMF-XL/2": {
        "sample_size": 32,
        "patch_size": 2,
        "hidden_size": 1024,
        "depth": 48,
        "num_attention_heads": 16,
        "aux_head_depth": 8,
    },
}


@dataclass
class IMFTransformer2DModelOutput(BaseOutput):
    velocity_u: torch.Tensor
    velocity_v: Optional[torch.Tensor] = None


def remap_legacy_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy iMF / Flax-style keys to native IMFTransformer2DModel keys."""
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("net.", "transformer."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        new_key = new_key.replace(".kernel", ".weight")
        remapped[new_key] = value
    return remapped


def config_from_legacy(config: Dict[str, object]) -> Dict[str, object]:
    model_type = config.get("model_type") or config.get("model_str") or config.get("model_name")
    if model_type not in IMF_PRESET_CONFIGS:
        raise ValueError(f"Unknown iMF preset '{model_type}'. Known: {list(IMF_PRESET_CONFIGS)}")

    preset = dict(IMF_PRESET_CONFIGS[model_type])
    preset["num_classes"] = int(config.get("num_class_embeds") or config.get("num_classes") or 1000)
    preset["model_type"] = model_type
    if config.get("sample_size") is not None:
        preset["sample_size"] = int(config["sample_size"])
    if config.get("in_channels") is not None:
        preset["in_channels"] = int(config["in_channels"])
    return preset


def _unsqueeze(t: torch.Tensor, dim: int) -> torch.Tensor:
    return t.unsqueeze(dim)


class _ScaledLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
        bias_init: str = "zeros",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if weight_init == "scaled_variance":
            std = init_constant / sqrt(in_features)
            nn.init.normal_(self.linear.weight, std=std)
        elif weight_init == "zeros":
            nn.init.zeros_(self.linear.weight)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")
        if bias:
            if bias_init == "zeros":
                nn.init.zeros_(self.linear.bias)
            else:
                raise ValueError(f"Invalid bias_init: {bias_init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if weight_init == "scaled_variance":
            std = init_constant / sqrt(embedding_dim)
            nn.init.normal_(self.embedding.weight, std=std)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        output = x * torch.rsqrt(mean_square + self.eps)
        return output.to(x.dtype) * self.weight


class _SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        init_kwargs = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = _ScaledLinear(in_features, hidden_features, **init_kwargs)
        self.w3 = _ScaledLinear(in_features, hidden_features, **init_kwargs)
        self.w2 = _ScaledLinear(hidden_features, in_features, **init_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class IMFTimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            _ScaledLinear(frequency_embedding_size, hidden_size, bias=True, init_constant=init_constant),
            nn.SiLU(),
            _ScaledLinear(hidden_size, hidden_size, bias=True, init_constant=init_constant),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class IMFLabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, init_constant: float = 1.0):
        super().__init__()
        self.embedding_table = _ScaledEmbedding(num_classes + 1, hidden_size, init_constant=init_constant)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class IMFPatchEmbedder(nn.Module):
    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        kh = kw = patch_size
        fan_in = kh * kw * in_channels
        fan_out = hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj.weight, -limit, limit)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0, device: torch.device | None = None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs_cis = torch.outer(positions, freqs)
    return torch.complex(torch.cos(freqs_cis), torch.sin(freqs_cis))


def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = x.to(torch.float32).view(torch.complex64)
    x_complex = x_complex.reshape(*x.shape[:-1], -1)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2).to(x.device)
    x_rotated = x_complex * freqs_cis
    x_out = x_rotated.to(x_complex.dtype).view(x.dtype)
    return x_out.reshape(x.shape)


class IMFRoPEAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, weight_init_constant: float = 0.32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        init_kwargs = dict(bias=False, init_constant=weight_init_constant)
        self.q_proj = _ScaledLinear(hidden_size, hidden_size, **init_kwargs)
        self.k_proj = _ScaledLinear(hidden_size, hidden_size, **init_kwargs)
        self.v_proj = _ScaledLinear(hidden_size, hidden_size, **init_kwargs)
        self.out_proj = _ScaledLinear(hidden_size, hidden_size, **init_kwargs)
        self.q_norm = _RMSNorm(self.head_dim)
        self.k_norm = _RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(self.q_norm(q), rope_freqs)
        k = apply_rotary_pos_emb(self.k_norm(k), rope_freqs)
        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        return self.out_proj(attn.reshape(batch, seq_len, self.hidden_size))


class IMFTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, weight_init_constant: float):
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = IMFRoPEAttention(hidden_size, num_heads, weight_init_constant=weight_init_constant)
        self.norm2 = _RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = _SwiGLUMlp(hidden_size, mlp_hidden_dim, weight_init_constant=weight_init_constant)
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class IMFFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = _RMSNorm(hidden_size)
        self.linear = _ScaledLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            init_constant=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class IMFTransformer2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        aux_head_depth: int = 8,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        eval_mode: bool = True,
        model_type: str | None = None,
        num_class_embeds: int | None = None,
    ):
        super().__init__()
        if num_class_embeds is not None:
            num_classes = int(num_class_embeds)
        if model_type in IMF_PRESET_CONFIGS:
            preset = IMF_PRESET_CONFIGS[model_type]
            sample_size = int(preset["sample_size"])
            patch_size = int(preset["patch_size"])
            hidden_size = int(preset["hidden_size"])
            depth = int(preset["depth"])
            num_attention_heads = int(preset["num_attention_heads"])
            aux_head_depth = int(preset["aux_head_depth"])

        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes
        self.aux_head_depth = aux_head_depth
        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens
        self.eval_mode = eval_mode
        self.gradient_checkpointing = False

        self.x_embedder = IMFPatchEmbedder(sample_size, patch_size, in_channels, hidden_size, bias=True)
        embed_kwargs = dict(hidden_size=hidden_size, init_constant=embedding_init_constant)
        self.h_embedder = IMFTimestepEmbedder(**embed_kwargs)
        self.omega_embedder = IMFTimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = IMFTimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = IMFTimestepEmbedder(**embed_kwargs)
        self.y_embedder = IMFLabelEmbedder(num_classes, hidden_size, init_constant=embedding_init_constant)

        token_std = token_init_constant / math.sqrt(hidden_size)
        self.time_tokens = nn.Parameter(torch.randn(num_time_tokens, hidden_size) * token_std)
        self.class_tokens = nn.Parameter(torch.randn(num_class_tokens, hidden_size) * token_std)
        self.omega_tokens = nn.Parameter(torch.randn(num_cfg_tokens, hidden_size) * token_std)
        self.t_min_tokens = nn.Parameter(torch.randn(num_interval_tokens, hidden_size) * token_std)
        self.t_max_tokens = nn.Parameter(torch.randn(num_interval_tokens, hidden_size) * token_std)

        total_tokens = (
            self.x_embedder.num_patches
            + num_class_tokens
            + num_cfg_tokens
            + 2 * num_interval_tokens
            + num_time_tokens
        )
        self.prefix_tokens = num_class_tokens + num_cfg_tokens + 2 * num_interval_tokens + num_time_tokens
        self.head_dim = hidden_size // num_attention_heads
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, total_tokens),
            persistent=False,
        )

        shared_depth = depth - aux_head_depth
        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            weight_init_constant=weight_init_constant,
        )
        self.shared_blocks = nn.ModuleList([IMFTransformerBlock(**block_kwargs) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([IMFTransformerBlock(**block_kwargs) for _ in range(aux_head_depth)])
        self.v_heads = nn.ModuleList(
            [IMFTransformerBlock(**block_kwargs) for _ in range(aux_head_depth if not eval_mode else 0)]
        )
        self.u_final_layer = IMFFinalLayer(hidden_size, patch_size, in_channels)
        self.v_final_layer = IMFFinalLayer(hidden_size, patch_size, in_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _build_sequence(
        self,
        sample: torch.Tensor,
        time_gap: torch.Tensor,
        guidance_scale: torch.Tensor,
        guidance_interval_start: torch.Tensor,
        guidance_interval_end: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.x_embedder(sample)
        h_embed = self.h_embedder(time_gap)
        omega_embed = self.omega_embedder(1 - 1 / guidance_scale)
        t_min_embed = self.cfg_t_start_embedder(guidance_interval_start)
        t_max_embed = self.cfg_t_end_embedder(guidance_interval_end)
        y_embed = self.y_embedder(class_labels)

        time_tokens = self.time_tokens + _unsqueeze(h_embed, 1)
        omega_tokens = self.omega_tokens + _unsqueeze(omega_embed, 1)
        t_min_tokens = self.t_min_tokens + _unsqueeze(t_min_embed, 1)
        t_max_tokens = self.t_max_tokens + _unsqueeze(t_max_embed, 1)
        class_tokens = self.class_tokens + _unsqueeze(y_embed, 1)

        return torch.cat(
            [class_tokens, omega_tokens, t_min_tokens, t_max_tokens, time_tokens, x_embed],
            dim=1,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        time_gap: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        guidance_interval_start: Optional[torch.Tensor] = None,
        guidance_interval_end: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[IMFTransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        batch_size = sample.shape[0]
        timestep = self._expand_batch(timestep, batch_size, sample.device, sample.dtype)
        if time_gap is None:
            time_gap = timestep
        else:
            time_gap = self._expand_batch(time_gap, batch_size, sample.device, sample.dtype)
        if guidance_scale is None:
            guidance_scale = torch.ones(batch_size, device=sample.device, dtype=sample.dtype)
        else:
            guidance_scale = self._expand_batch(guidance_scale, batch_size, sample.device, sample.dtype)
        if guidance_interval_start is None:
            guidance_interval_start = torch.zeros(batch_size, device=sample.device, dtype=sample.dtype)
        else:
            guidance_interval_start = self._expand_batch(
                guidance_interval_start, batch_size, sample.device, sample.dtype
            )
        if guidance_interval_end is None:
            guidance_interval_end = torch.ones(batch_size, device=sample.device, dtype=sample.dtype)
        else:
            guidance_interval_end = self._expand_batch(
                guidance_interval_end, batch_size, sample.device, sample.dtype
            )

        class_labels = class_labels.reshape(-1).long()
        seq = self._build_sequence(
            sample,
            time_gap,
            guidance_scale,
            guidance_interval_start,
            guidance_interval_end,
            class_labels,
        )

        for block in self.shared_blocks:
            if self.training and self.gradient_checkpointing:
                seq = torch.utils.checkpoint.checkpoint(block, seq, self.rope_freqs, use_reentrant=False)
            else:
                seq = block(seq, self.rope_freqs)

        u_seq = v_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)
        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)

        u_tokens = u_seq[:, self.prefix_tokens :]
        velocity_u = self.unpatchify(self.u_final_layer(u_tokens))

        velocity_v = None
        if len(self.v_heads) > 0:
            v_tokens = v_seq[:, self.prefix_tokens :]
            velocity_v = self.unpatchify(self.v_final_layer(v_tokens))

        if not return_dict:
            if velocity_v is None:
                return (velocity_u,)
            return (velocity_u, velocity_v)
        return IMFTransformer2DModelOutput(velocity_u=velocity_u, velocity_v=velocity_v)

    @staticmethod
    def _expand_batch(
        value: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.reshape(1).repeat(batch_size)
        else:
            value = value.reshape(-1)
            if value.shape[0] == 1 and batch_size > 1:
                value = value.repeat(batch_size)
        return value

    @classmethod
    def from_imf_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str | None = None,
        map_location: str = "cpu",
        strict: bool = True,
        eval_mode: bool = True,
    ) -> Tuple["IMFTransformer2DModel", Dict[str, object]]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, Mapping) and any(k.startswith("net.") for k in checkpoint):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        inferred_type = model_type
        if inferred_type is None and isinstance(checkpoint, Mapping):
            args = checkpoint.get("args")
            if isinstance(args, argparse.Namespace):
                inferred_type = getattr(args, "model", None) or getattr(args, "model_str", None)
            elif isinstance(args, Mapping):
                inferred_type = args.get("model") or args.get("model_str")

        if inferred_type is None:
            raise ValueError("model_type must be provided when loading a raw state dict checkpoint.")

        config = dict(IMF_PRESET_CONFIGS[inferred_type])
        config["model_type"] = inferred_type
        config["eval_mode"] = eval_mode
        model = cls(**config)
        model.load_state_dict(remap_legacy_state_dict(state_dict), strict=strict)
        metadata = {"checkpoint_path": checkpoint_path, "model_type": inferred_type}
        return model, metadata

    def to_imf_checkpoint(self, prefix: str = "net.") -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for key, value in self.state_dict().items():
            if key == "rope_freqs":
                continue
            state[f"{prefix}{key}"] = value.detach().cpu()
        return state

    @property
    def net(self):
        return self


IMFDiffusersModel = IMFTransformer2DModel
