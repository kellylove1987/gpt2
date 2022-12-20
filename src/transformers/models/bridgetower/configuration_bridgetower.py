# coding=utf-8
# Copyright 2022 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BridgeTower model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
    "BridgeTower/bridgetower-base-itm-mlm": (
        "https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm/blob/main/config.json"
    ),
}


class BridgeTowerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BridgeTowerModel`]. It is used to instantiate a
    BridgeTower model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        cross_modal_transform_shared (`bool`, *optional*, defaults to `True`):
            Whether cross modal transformer layers are shared.
        drop_rate (`float`, *optional*, defaults to 0.1):
            Drop out probability.
        freeze_roberta (`bool`, *optional*, defaults to `False`):
            Whether to freeze roberta.
        freeze_vit (`bool`, *optional*, defaults to `False`):
            Whether to freeze vit.
        freeze_layer_count_roberta (`bool`, *optional*, defaults to `False`):
            Whether to freeze layer count for RobERTa.
        freeze_layer_count_vit (`bool`, *optional*, defaults to `False`):
            Whether to freeze layer count for vit.
        head_hidden_scale (`int`, *optional*, defaults to 2):
            Scale of hidden layers head.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        image_size (`int`, *optional*, defaults to 288):
            The size (resolution) of each image.
        input_image_embed_size (`int`, *optional*, defaults to 768):
            Embedding size of the input image.
        input_text_embed_size (`int`, *optional*, defaults to 768):
            Embedding size of the input text.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether this is an encoder/decoder model
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        link_tower_shared (`bool`, *optional*, defaults to `False`):
            Whether the bride/link tower layers are shared.
        link_tower_type (`str`, *optional*, defaults to `"add"`):
            Type of the bridge/link layer.
        max_text_len (`int`, *optional*, defaults to 50):
            Maximum text length.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of MLP hidden dim to embedding dim.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        resolution_before (`int`, *optional*, defaults to 224):
            Prior resolution.
        stop_gradient (`bool`, *optional*, defaults to `False`):
            Whether to stop gradient for training.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether embedding weights are tied with the decoder
        tokenizer (`str`, *optional*, defaults to `"roberta-base"`):
            Choice of the text tokenizer.
        unfreeze_roberta_attention (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze roberta's LayerNorm.
        unfreeze_roberta_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze roberta's embeddings.
        unfreeze_roberta_encoder (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze roberta's encoder.
        unfreeze_roberta_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze roberta's LayerNorm.
        unfreeze_vit_attention (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze vit's attention.
        unfreeze_vit_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to unfreeze vit's LayerNorm.
        vit_embed_dim (`int`, *optional*, defaults to 512):
            Dimension size of embeddings in vit model.
        vit_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        vit_num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in vit model.
        vit_layernorm_init_from_vit (`bool`, *optional*, defaults to `False`):
            Whether to init vit LayerNorm from vit.
        vit_layernorm_shared (`bool`, *optional*, defaults to `True`):
            Whether vit's LayerNorm layers are shared.
        vit_patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch in vit.
        vit_remove_last (`bool`, *optional*, defaults to `False`):
            Whether to remove vit's last layer.
        vit_intermediate_size (`int`, *optional*, defaults to 512):
            Dimension of vit's transformer intermediate layer.
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the text part of the model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`BridgeTowerModel`].

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bridgetower"

    def __init__(
        self,
        cross_modal_transform_shared=True,
        drop_rate=0.1,
        freeze_roberta=False,
        freeze_vit=False,
        freeze_layer_count_roberta=False,
        freeze_layer_count_vit=False,
        head_hidden_scale=2,
        hidden_act="gelu",
        hidden_size=768,
        image_size=288,
        input_image_embed_size=768,
        input_text_embed_size=768,
        is_encoder_decoder=False,
        layer_norm_eps=1e-05,
        link_tower_shared=False,
        link_tower_type="add",
        max_text_len=50,
        mlp_ratio=4,
        num_attention_heads=12,
        num_hidden_layers=6,
        resolution_before=224,
        stop_gradient=False,
        tie_word_embeddings=False,
        tokenizer="roberta-base",
        unfreeze_roberta_attention=False,
        unfreeze_roberta_embeddings=False,
        unfreeze_roberta_encoder=False,
        unfreeze_roberta_layernorm=False,
        unfreeze_vit_attention=False,
        unfreeze_vit_layernorm=False,
        vit_embed_dim=512,
        vit_num_hidden_layers=12,
        vit_layernorm_init_from_vit=False,
        vit_layernorm_shared=True,
        vit_patch_size=16,
        vit_remove_last=False,
        vit_intermediate_size=512,
        vit_hidden_size=768,
        vocab_size=50265,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cross_modal_transform_shared = cross_modal_transform_shared
        self.drop_rate = drop_rate
        self.freeze_roberta = freeze_roberta
        self.freeze_vit = freeze_vit
        self.freeze_layer_count_roberta = freeze_layer_count_roberta
        self.freeze_layer_count_vit = freeze_layer_count_vit
        self.head_hidden_scale = head_hidden_scale
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.input_image_embed_size = input_image_embed_size
        self.input_text_embed_size = input_text_embed_size
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_eps = layer_norm_eps
        self.link_tower_shared = link_tower_shared
        self.link_tower_type = link_tower_type
        self.max_text_len = max_text_len
        self.mlp_ratio = mlp_ratio
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.resolution_before = resolution_before
        self.stop_gradient = stop_gradient
        self.tie_word_embeddings = tie_word_embeddings
        self.tokenizer = tokenizer
        self.unfreeze_roberta_attention = unfreeze_roberta_attention
        self.unfreeze_roberta_embeddings = unfreeze_roberta_embeddings
        self.unfreeze_roberta_encoder = unfreeze_roberta_encoder
        self.unfreeze_roberta_layernorm = unfreeze_roberta_layernorm
        self.unfreeze_vit_attention = unfreeze_vit_attention
        self.unfreeze_vit_layernorm = unfreeze_vit_layernorm
        self.vit_embed_dim = vit_embed_dim
        self.vit_num_hidden_layers = vit_num_hidden_layers
        self.vit_layernorm_init_from_vit = vit_layernorm_init_from_vit
        self.vit_layernorm_shared = vit_layernorm_shared
        self.vit_patch_size = vit_patch_size
        self.vit_remove_last = vit_remove_last
        self.vit_intermediate_size = vit_intermediate_size
        self.vit_hidden_size = vit_hidden_size
        self.vocab_size = vocab_size
