# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
""" Swin Backbone model for OneFormer."""
import collections.abc
import math
from dataclasses import dataclass
from typing import  Optional, Tuple
import torch.nn.functional as F
import torch
from torch import  nn
from transformers.utils import logging
from ...modeling_utils import ModuleUtilsMixin
from ...activations import ACT2FN
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput

from .configuration_oneformer import OneFormerConfig
logger = logging.get_logger(__name__)

################## Utilities #################

# Copied from transformers.models.maskformer.modeling_maskformer_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.maskformer.modeling_maskformer_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.maskformer.modeling_maskformer_swin.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

################## Data Classes #################
    
# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinModelOutputWithPooling with Mask->One
@dataclass
class OneFormerSwinModelOutput(ModelOutput):
    """
    Class for OneFormerSwinModel's outputs

    Args:
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot be inferred before the
            `forward` method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None   


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinBaseModelOutput with Mask->One
@dataclass
class OneFormerSwinBaseModelOutput(ModelOutput):
    """
    Class for SwinEncoder's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot inferred before the `forward`
            method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

################## Swin Backbone Classes #################

# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinEmbeddings with Mask->One
class OneFormerSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = OneFormerSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size

        if config.backbone_config["use_absolute_embeddings"]:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.backbone_config["embed_dim"]))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.backbone_config["embed_dim"])
        self.dropout = nn.Dropout(config.backbone_config["hidden_dropout_prob"])

    def forward(self, pixel_values):
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions
    
# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinPatchEmbeddings with Mask->One
class OneFormerSwinPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding, including padding.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.backbone_config["image_size"], config.backbone_config["patch_size"]
        num_channels, hidden_size = config.backbone_config["num_channels"], config.backbone_config["embed_dim"]

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = F.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = F.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values):
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings_flat = embeddings.flatten(2).transpose(1, 2)

        return embeddings_flat, output_dimensions


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinPatchMerging with Mask->One
class OneFormerSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer for oneformer model.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, width, height):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = F.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature, input_dimensions):
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinDropPath with Mask->One
class OneFormerSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinSelfAttention with Mask->One
class OneFormerSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.backbone_config["qkv_bias"])

        self.dropout = nn.Dropout(config.backbone_config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in MaskFormerSwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinSelfOutput with Mask->One
class OneFormerSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.backbone_config["attention_probs_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinAttention with Mask->One
class OneFormerSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        self.self = OneFormerSwinSelfAttention(config, dim, num_heads, window_size)
        self.output = OneFormerSwinSelfOutput(config, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinIntermediate with Mask->One
class OneFormerSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, int(config.backbone_config["mlp_ratio"] * dim))
        if isinstance(config.backbone_config["hidden_act"], str):
            self.intermediate_act_fn = ACT2FN[config.backbone_config["hidden_act"]]
        else:
            self.intermediate_act_fn = config.backbone_config["hidden_act"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    
# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinOutput with Mask->One
class OneFormerSwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.backbone_config["mlp_ratio"] * dim), dim)
        self.dropout = nn.Dropout(config.backbone_config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    
# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinIntermediate with Mask->One
class OneFormerSwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.backbone_config["window_size"]
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.general_config["layer_norm_eps"])
        self.attention = OneFormerSwinAttention(config, dim, num_heads, self.window_size)
        self.drop_path = (
            OneFormerSwinDropPath(config.backbone_config["drop_path_rate"]) if config.backbone_config["drop_path_rate"] > 0.0 else nn.Identity()
        )
        self.layernorm_after = nn.LayerNorm(dim, eps=config.general_config["layer_norm_eps"])
        self.intermediate = OneFormerSwinIntermediate(config, dim)
        self.output = OneFormerSwinOutput(config, dim)

    def get_attn_mask(self, input_resolution):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            height, width = input_resolution
            img_mask = torch.zeros((1, height, width, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_left = pad_top = 0
        pad_rigth = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, pad_left, pad_rigth, pad_top, pad_bottom)
        hidden_states = F.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
        height, width = input_dimensions
        batch_size, dim, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask((height_pad, width_pad))
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        self_attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )  # B height' width' C

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        outputs = (layer_output,) + outputs

        return outputs

    
# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinStage with Mask->One
class OneFormerSwinStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList(
            [
                OneFormerSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.backbone_config["window_size"] // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        all_hidden_states = () if output_hidden_states else None

        height, width = input_dimensions
        for i, block_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            block_hidden_states = block_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = block_hidden_states[0]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        return hidden_states, output_dimensions, all_hidden_states


# Copied from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinEncoder with Mask->One
class OneFormerSwinEncoder(nn.Module):
    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.backbone_config["depths"])
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.backbone_config["drop_path_rate"], sum(config.backbone_config["depths"]))]
        self.layers = nn.ModuleList(
            [
                OneFormerSwinStage(
                    config=config,
                    dim=int(config.backbone_config["embed_dim"] * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.backbone_config["depths"][i_layer],
                    num_heads=config.backbone_config["num_heads"][i_layer],
                    drop_path=dpr[sum(config.backbone_config["depths"][:i_layer]) : sum(config.backbone_config["depths"][: i_layer + 1])],
                    downsample=OneFormerSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False
        self.training = config.general_config["is_train"]

    def forward(
        self,
        hidden_states,
        input_dimensions,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_input_dimensions = ()
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_hidden_states, output_dimensions, layer_all_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states, layer_head_mask
                )
            else:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    output_hidden_states,
                )

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                all_hidden_states += (layer_all_hidden_states,)

            hidden_states = layer_hidden_states

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_all_hidden_states[1],)

        return OneFormerSwinBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            hidden_states_spatial_dimensions=all_input_dimensions,
            attentions=all_self_attentions,
        )

    
# Modified from transformers.models.maskformer.modeling_maskformer_swin.MaskFormerSwinModel with Mask->One
class OneFormerSwinModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: OneFormerConfig, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.num_layers = len(config.backbone_config["depths"])
        self.num_features = int(config.backbone_config["embed_dim"] * 2 ** (self.num_layers - 1))

        self.embeddings = OneFormerSwinEmbeddings(config)
        self.encoder = OneFormerSwinEncoder(config, self.embeddings.patch_grid)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.backbone_config["depths"]))

        embedding_output, input_dimensions = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states_spatial_dimensions = (input_dimensions,) + encoder_outputs.hidden_states_spatial_dimensions

        if not return_dict:
            output = (encoder_outputs.hidden_states, hidden_states_spatial_dimensions,)
            if output_attentions:
                output += (encoder_outputs.attentions,)
            return output

        return OneFormerSwinModelOutput(
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_spatial_dimensions=hidden_states_spatial_dimensions,
            attentions=encoder_outputs.attentions,
        )