# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" PyTorch ProphetNet model, ported from ProphetNet repo(fairsequery_states version). """

import copy
import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ACT2FN
from .configuration_prophetnet import ProphetNetConfig
from .file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_outputs import BaseModelOutput
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "ProphetNetTokenizer"

PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/prophetnet-large-uncased",
    # See all ProphetNet models at https://huggingface.co/models?filter=prophetnet
]


PROPHETNET_START_DOCSTRING = r"""
    Model and checkpoints are converted from ProphetNet and xProphetNet original Fairsequery_states version repo.
    Details can be found from <https://github.com/microsoft/ProphetNet>
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.ProphetNetConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""
PROPHETNET_GENERATION_EXAMPLE = r"""
    ProphetNet Summarization example::

        from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')

        ARTICLE_TO_SUMMARIZE = "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a high-level science and technology workforce, as deemed critical for development of China's economy, defense, and science and technology education. The establishment was hailed as \"A Major Event in the History of Chinese Education and Science.\" CAS has supported USTC by combining most of its institutes with the departments of the university. USTC is listed in the top 16 national key universities, becoming the youngest national key university.".lower()
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=100, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
        print([tokenizer.decode(g) for g in summary_ids])
    xProphetNet xGLUE News Title Generation example:
        from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')

        EN_SENTENCE = "Microsoft Corporation intends to officially end free support for the Windows 7 operating system after January 14, 2020, according to the official portal of the organization. From that day, users of this system will not be able to receive security updates, which could make their computers vulnerable to cyber attacks."
        RU_SENTENCE = "орпорация Microsoft намерена официально прекратить бесплатную поддержку операционной системы Windows 7 после 14 января 2020 года, сообщается на официальном портале организации . С указанного дня пользователи этой системы не смогут получать обновления безопасности, из-за чего их компьютеры могут стать уязвимыми к кибератакам."
        ZH_SENTENCE = "根据该组织的官方门户网站，微软公司打算在2020年1月14日之后正式终止对Windows 7操作系统的免费支持。从那时起，该系统的用户将无法接收安全更新，这可能会使他们的计算机容易受到网络攻击。"
        inputs = tokenizer([EN_SENTENCE, RU_SENTENCE, ZH_SENTENCE], padding=True, max_length=256, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        print([tokenizer.decode(g) for g in summary_ids])
"""

PROPHETNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use ProphetNetTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.ProphetNetTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
"""


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def ngram_attention_bias(sequence_length, ngram, device, dtype):
    """
    This function computes the bias for the predict stream
    """
    bias = torch.ones((ngram, sequence_length, 2 * sequence_length), device=device, dtype=dtype) * float("-inf")
    # create bias
    for stream_idx in range(ngram):
        for i in range(sequence_length):
            bias[stream_idx, i, sequence_length + i] = 0
            bias[stream_idx, i, : max(i - stream_idx, 0) + 1] = 0
    return bias


def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = (
            rel_positions_bucket
            + torch.lt(inv_relative_positions, torch.zeros_like(inv_relative_positions)).int() * num_buckets
        )
        inv_relative_positions = torch.abs(inv_relative_positions)
    else:
        inv_relative_positions = torch.max(inv_relative_positions, torch.zeros_like(inv_relative_positions))

    max_exact = num_buckets // 2
    is_small = torch.lt(inv_relative_positions, max_exact)
    val_if_large = max_exact + torch.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + torch.where(is_small, inv_relative_positions.int(), val_if_large)
    return rel_positions_bucket


def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # main stream
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.size(-1), 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # predicting stream
    predicting_stream_relative_positions = torch.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.size(-1), 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # get both position buckets
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    return main_relative_position_buckets, predict_relative_position_buckets


@dataclass
class ProphetNetSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the main stream language modeling head (scores for each vocabulary token before SoftMax).
        logits_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Prediction scores of the predict stream language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the weighted average in the
        decoder_cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to compute the weighted average in the
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ProphetNetSeq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`):
            Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        last_hidden_state_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the weighted average in the
        decoder_cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to compute the weighted average in the
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ProphetNetDecoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`):
            Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.

            If :obj:`past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        last_hidden_state_ngram (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, ngram * decoder_sequence_length, config.vocab_size)`):
            Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, decoder_sequence_length, hidden_size)`.

            Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.
        ngram_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, ngram * decoder_sequence_length, hidden_size)`.

            Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        ngram_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the weighted average in the
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)`.

            Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to compute the weighted average in the
    """

    last_hidden_state: torch.FloatTensor
    last_hidden_state_ngram: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_ngram: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ngram_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class ProphetNetPreTrainedModel(PreTrainedModel):
    config_class = ProphetNetConfig
    base_model_prefix = "prophetnet"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In ProphetNet it is usually set to the pad_token_id. See ProphetNet docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, config: ProphetNetConfig):
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)

    def forward(self, inputs_shape, device, attention_mask=None, past_key_values=None, position_ids=None):
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        if position_ids is None:
            if past_key_values is not None:
                # position_ids is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                prev_num_input_ids = past_key_values[0]["self"]["prev_key_states"].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                position_ids = torch.ones((1, 1), dtype=torch.long, device=device) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                if attention_mask is None:
                    attention_mask = torch.ones(inputs_shape, dtype=torch.long, device=device)

                # retrieve position_ids from input_ids / attention_mask
                position_ids = (
                    torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

        return super().forward(position_ids), position_ids

    def _forward(self, position_ids):
        return super().forward(position_ids)


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: ProphetNetConfig,
        num_attn_heads: int,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.num_attn_heads = num_attn_heads
        self.head_dim = hidden_size // num_attn_heads

        assert (
            self.head_dim * num_attn_heads == hidden_size
        ), "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and `config.num_decoder_attention_heads`"

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _reshape(self, tensor, first_dim, batch_size):
        return tensor.reshape(first_dim, batch_size * self.num_attn_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        sequence_length, batch_size, hidden_size = hidden_states.size()

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        cache_key = "cross_attention" if is_cross_attention else "self"
        assert list(hidden_states.size()) == [
            sequence_length,
            batch_size,
            hidden_size,
        ], f"Size of hidden states should be {sequence_length, batch_size, hidden_size}, but is {hidden_states.size()}"

        # previous time steps are cached - no need to recompute key and value if they are static
        if layer_state is not None:
            saved_state = layer_state.get(cache_key, None)

        query_states = self.query_proj(hidden_states) / (self.head_dim ** 0.5)
        query_states = self._reshape(query_states, sequence_length, batch_size)

        if not is_cross_attention:
            # self-attention
            key_states = self.key_proj(hidden_states)
            key_states = self._reshape(key_states, -1, batch_size)
            value_states = self.value_proj(hidden_states)
            value_states = self._reshape(value_states, -1, batch_size)
        elif saved_state is None:
            # cross-attention without layer state
            key_states = self.key_proj(key_value_states)
            key_states = self._reshape(key_states, -1, batch_size)
            value_states = self.value_proj(key_value_states)
            value_states = self._reshape(value_states, -1, batch_size)
        else:
            key_states = saved_state["prev_key_states"].view(batch_size * self.num_attn_heads, -1, self.head_dim)
            value_states = saved_state["prev_value_states"].view(batch_size * self.num_attn_heads, -1, self.head_dim)

        # Update cache
        if is_cross_attention:
            layer_state[cache_key] = {
                "prev_key_states": key_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
                "prev_value_states": value_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
            }

        key_sequence_length = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        assert attn_weights.size() == (batch_size * self.num_attn_heads, sequence_length, key_sequence_length), f"`attn_weights` should be of size {batch_size * self.num_attn_heads, sequence_length, key_sequence_length}, but is of size {attn_weights.shape}"

        if attn_mask is not None:
            attn_weights = (
                attn_weights.view(batch_size, self.num_attn_heads, sequence_length, key_sequence_length) + attn_mask
            )
            attn_weights = attn_weights.view(batch_size * self.num_attn_heads, sequence_length, key_sequence_length)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if attention_mask is not None and attention_mask.dim() == 0:
            attention_mask = None
        assert attention_mask is None or attention_mask.size() == (
            self.num_attn_heads * batch_size,
            1,
            key_sequence_length,
        ), f"`attention_mask` should be `None` or of shape attention_mask.size() == {batch_size * self.num_attn_heads, 1, key_sequence_length}, but is {attention_mask.shape}"

        if attention_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (
            batch_size * self.num_attn_heads,
            sequence_length,
            self.head_dim,
        ), "`attn_output` should be of shape {batch_size * self.num_attn_heads, sequence_length, self.head_dim}, but is of shape {attn_output.size()}"
        attn_output = attn_output.transpose(0, 1).contiguous().view(sequence_length, batch_size, hidden_size)

        attn_output = self.out_proj(attn_output)

        if output_attentions:
            attn_weights = attn_weights.view(batch_size, self.num_attn_heads, sequence_length, key_sequence_length)
        else:
            attn_weights = None

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        return attn_output, attn_weights


class FeedForwardBlock(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original
    Transformer implementation.
    """

    def __init__(self, config: ProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)
        self.output = nn.Linear(ffn_dim, config.hidden_size)
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.output(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class NgramMultiheadAttention(nn.Module):
    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.num_attn_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        self.head_dim = config.hidden_size // self.num_attn_heads
        self.ngram = config.ngram

        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        # key, value, query projection
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # out projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # rel position embeddings
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)

        # for onnx runtime
        self.onnx_trace = False

    def _reshape(self, tensor, first_dim, batch_size):
        return tensor.reshape(first_dim, batch_size * self.num_attn_heads, self.head_dim).transpose(0, 1)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        hidden_states,
        layer_state=None,
        need_weights=True,
        attention_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
        sequence_length, batch_size, hidden_size = hidden_states.size()

        assert list(hidden_states.size()) == [sequence_length, batch_size, hidden_size], f"`hidden_states` should be of shape {sequence_length, batch_size, hidden_size}, but is of shape {hidden_states.shape}"

        # key and value of previous time steps are cached
        saved_state = layer_state.get("self", None)

        # project
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)

        # normalize
        query_states = query_states / (self.head_dim ** 0.5)

        # reshape
        query_states = self._reshape(query_states, sequence_length, batch_size)
        key_states = self._reshape(key_states, -1, batch_size)
        value_states = self._reshape(value_states, -1, batch_size)

        # chunk into main stream and predict stream
        hidden_states_list = hidden_states.chunk(1 + self.ngram, dim=0)

        query_states_list = query_states.chunk(1 + self.ngram, dim=1)
        key_states_list = key_states.chunk(1 + self.ngram, dim=1)
        value_states_list = value_states.chunk(1 + self.ngram, dim=1)

        main_hidden_states, hidden_states_predict_list = hidden_states_list[0], hidden_states_list[1:]
        main_query_states, predict_query_states_list = query_states_list[0], query_states_list[1:]
        main_key_states, predict_key_states_list = key_states_list[0], key_states_list[1:]
        main_value_states, predict_value_states_list = value_states_list[0], value_states_list[1:]

        # saved states are stored with shape (batch_size, num_attn_heads, seq_len, head_dim)
        if saved_state is not None:
            prev_main_key_states = saved_state["prev_key_states"].view(
                batch_size * self.num_attn_heads, -1, self.head_dim
            )
            main_key_states = torch.cat((prev_main_key_states, main_key_states), dim=1)
            prev_main_value_states = saved_state["prev_value_states"].view(
                batch_size * self.num_attn_heads, -1, self.head_dim
            )
            main_value_states = torch.cat((prev_main_value_states, main_value_states), dim=1)

        # Update cache
        layer_state["self"] = {
            "prev_key_states": main_key_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
            "prev_value_states": main_value_states.view(batch_size, self.num_attn_heads, -1, self.head_dim),
        }

        # get seq_length of main stream only
        main_sequence_length = sequence_length // (1 + self.ngram)

        # MAIN-STREAM
        # main attn weights
        main_attn_weights = torch.bmm(main_query_states, main_key_states.transpose(1, 2))

        # retrieve relative position embeddings for each layer -> see paper for more details
        main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(
            main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets
        )
        main_attn_weights = main_attn_weights + main_relative_pos_embeddings

        if attention_mask is not None:
            main_attn_weights = main_attn_weights + attention_mask

        main_attn_probs = softmax(
            main_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(main_attn_weights)

        main_attn_probs = F.dropout(main_attn_probs, p=self.attention_dropout, training=self.training)

        # project to attn_output
        main_attn_output = torch.bmm(main_attn_probs, main_value_states)
        main_attn_output = (
            main_attn_output.transpose(0, 1).contiguous().view(1, main_sequence_length, batch_size, hidden_size)
        )
        main_attn_output = self.out_proj(main_attn_output)

        # PREDICT-STREAM
        # [ngram, B*head, T, c]
        predict_query_states = torch.cat(predict_query_states_list, 0).view(
            self.ngram, -1, main_sequence_length, self.head_dim
        )
        # [ngram, B*head, 2*T, c]
        predict_key_states = torch.cat(
            [torch.cat([main_key_states, key], 1).unsqueeze(0) for key in predict_key_states_list], 0
        )

        # [ngram, T, B, C]
        predict_hidden_states = torch.cat(hidden_states_predict_list, 0).view(
            self.ngram, main_sequence_length, batch_size, hidden_size
        )

        # [ngram, B*head, 2*T, c]
        predict_value_states = torch.cat(
            [torch.cat([main_value_states, v_p], 1).unsqueeze(0) for v_p in predict_value_states_list], 0
        )
        # [ngram, B*head, T, 2*T]
        predict_attn_weights = torch.einsum("nbtc,nbsc->nbts", (predict_query_states, predict_key_states))

        # [ngram, B*head, T, S]
        # retrieve relative position embeddings for each layer -> see paper for more details
        predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
            predict_hidden_states, predict_attn_weights, position_ids, predict_relative_position_buckets
        )

        # [ngram, B*head, T, 2*T]
        predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings

        if extended_predict_attention_mask is not None:
            predict_attn_weights = predict_attn_weights + extended_predict_attention_mask

        predict_attn_probs = softmax(
            predict_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(predict_attn_weights)
        predict_attn_probs = F.dropout(predict_attn_probs, p=self.attention_dropout, training=self.training)

        # project to attention output
        # [ngram, B*head, T, c]
        predict_attn_output = torch.einsum("nbts,nbsc->nbtc", (predict_attn_probs, predict_value_states))
        # [ngram, T, B, C]
        predict_attn_output = (
            predict_attn_output.transpose(1, 2)
            .contiguous()
            .view(self.ngram, main_sequence_length, batch_size, hidden_size)
        )
        predict_attn_output = self.out_proj(predict_attn_output)

        # concat to single attn output
        # [1+ngram*T, B, C]
        attn_output = torch.cat([main_attn_output, predict_attn_output], 0).view(-1, batch_size, hidden_size)

        # reshape into better form for `config.output_attentions`
        main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, main_sequence_length, -1)
        predict_attn_probs = predict_attn_probs.view(
            self.ngram, batch_size, self.num_attn_heads, main_sequence_length, -1
        ).transpose(0, 1)

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        return attn_output, main_attn_probs, predict_attn_probs

    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hidden_states [T,B,C], input attn_weights [T*head,T,S], input position_ids [B,T] or [1,1]

        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            relative_positions = (
                torch.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(
                batch_size, sequence_length, 1
            )  # [B, T, s]
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        hidden_states = hidden_states.transpose(0, 1)  # [B,T,C]
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)  # [B,T,Buckets*head]
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        ).permute(
            0, 3, 1, 2
        )  # [B,T,Buckets,head]
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:2] + (-1,))  # [B*head,T,Buckets]

        main_relative_position_buckets = (
            main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
            .view(-1, main_relative_position_buckets.shape[-1])
            .long()
        )  # [B*head*T, T]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))  # [B*head*T,Buckets]

        main_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=main_relative_position_buckets
        ).view(attn_weights.shape[:2] + (-1,))

        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
    ):
        # input hidden_states [ngram, T,B,C], input attn_weights [ngram, B*head,T,S], input position_ids [B,T] or [1,1], input predict_relative_position_buckets [B,T, 2*T] or None

        sequence_length, batch_size = hidden_states.shape[1:3]

        if predict_relative_position_buckets is None:
            key_sequence_length = attn_weights.shape[-1]
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            relative_positions = (
                torch.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
                .to(position_ids.device)
            )

            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        hidden_states = hidden_states.transpose(1, 2)  # [ngram, B, T, C]
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states).view(
            hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )  # [ngram, B, T, bucket, head]
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 1, 4, 2, 3).reshape(
            self.ngram * batch_size * self.num_attn_heads, sequence_length, -1
        )  # [ngram*B*head, T, bucket]

        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0).repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )  # [ngram, B, head*T, S]

        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.size(-1)
        ).long()  # [ngram*B*head*T, S]

        predict_relative_pos_embeddings = torch.gather(
            rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
        ).view(
            self.ngram, batch_size * self.num_attn_heads, sequence_length, -1
        )  # [ngram, B*head, T, S]

        return predict_relative_pos_embeddings


class ProphetNetEncoderLayer(nn.Module):
    """
        Encoder block for Prophetnet
    """
    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        # 1st residual block
        self.self_attn = SelfAttention(config, config.num_encoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        self.feed_forward = FeedForwardBlock(config, config.encoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask, output_attentions=False):
        # 1st residual block
        attention_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)
        return hidden_states, attn_weights


class ProphetNetDecoderLayer(nn.Module):
    """
        Decoder block for Prophetnet
    """
    def __init__(self, config: ProphetNetConfig):
        super().__init__()
        # 1st residual block
        self.self_attn = NgramMultiheadAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)

        # 2nd residual block
        self.cross_attn = SelfAttention(config, config.num_decoder_attention_heads)
        self.cross_attn_layer_norm = LayerNorm(config.hidden_size)

        # 3rd residual block
        self.feed_forward = FeedForwardBlock(config, config.decoder_ffn_dim)
        self.feed_forward_layer_norm = LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        layer_state=None,
        attention_mask=None,
        output_attentions=False,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
        layer_state = layer_state if layer_state is not None else {}

        # 1st residual block
        ngram_attention_output, self_attn_weights, self_attn_weights_ngram = self.self_attn(
            hidden_states=hidden_states,
            layer_state=layer_state,
            need_weights=False,
            attention_mask=attention_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 2nd residual block
            attention_output, cross_attn_weights = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attn_mask,
                layer_state=layer_state,  # mutates layer state
            )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

        # 3rd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        return (
            hidden_states,
            self_attn_weights,
            self_attn_weights_ngram,
            cross_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class ProphetNetEncoder(ProphetNetPreTrainedModel):
    """
    Same to Transformer Encoder.
    Transformer encoder consisting of *config.num_encoder_layers* self attention layers. Each layer
    is a :class:`ProphetNetEncoderLayer`.

    Args:
        config: ProphetNetConfig
    """

    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = None):
        super().__init__(config)

        self.word_embeddings = word_embeddings
        self.position_embeddings = LearnedPositionalEmbedding(config)
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        self.layers = nn.ModuleList([ProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # prepare attention mask
        if attention_mask is not None:
            extended_attention_mask = (
                1.0 - attention_mask[:, None, :].repeat(self.config.num_attention_heads, 1, 1)
            ) * -10000.0
        else:
            extended_attention_mask = None

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds has to be passed.")
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("Make sure to only pass input_ids or inputs_embeds.")
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings, position_ids = self.position_embeddings(inputs_embeds.shape[:2], inputs_embeds.device)

        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        hidden_states = hidden_states.transpose(0, 1)  # B x T x C -> T x B x C

        encoder_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_hidden_states = encoder_hidden_states + (hidden_states.transpose(0, 1),)
            hidden_states, attn_probs = encoder_layer(
                hidden_states, attention_mask=extended_attention_mask, output_attentions=output_attentions
            )
            if output_attentions:
                all_attentions = all_attentions + (attn_probs,)

        hidden_states = hidden_states.transpose(0, 1)
        if output_hidden_states:
            encoder_hidden_states = encoder_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_hidden_states, attentions=all_attentions
        )


class ProphetNetDecoder(ProphetNetPreTrainedModel):
    """
    N-stream decoder. One main stream, self.ngram predicting streams.
    Next self.ngram tokens are predicted.

    N-stream decoder consisting of *config.num_decoder_layers* layers. Each layer
    is a :class:`ProphetNetDecoderLayer`.
    Args:
        config: ProphetNetConfig
        word_embeddings (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: ProphetNetConfig, word_embeddings: nn.Embedding = None):
        super().__init__(config)

        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance

        self.dropout = config.dropout
        self.padding_idx = word_embeddings.padding_idx
        self.max_target_positions = config.max_position_embeddings

        self.word_embeddings = word_embeddings
        self.position_embeddings = LearnedPositionalEmbedding(config)

        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        self.layers = nn.ModuleList([ProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, sequence_length)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_attention_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation
            use_cache: inference or training procedure.

        Returns:
            tuple:
                - the decoder's features of next n-grams, with shape `(batch, self.ngram * sequence_length, hidden_size)`
                - hidden states
                - attentions

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `decoder_input_ids` or `decoder_inputs_embeds` has to be passed.")
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("Make sure to only pass `decoder_input_ids` or `decoder_inputs_embeds`.")
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]

        main_stream_pos_embed, position_ids = self.position_embeddings(
            (batch_size, sequence_length),
            device=inputs_embeds.device,
            past_key_values=past_key_values,
        )

        if past_key_values is not None:
            main_relative_position_buckets, predict_relative_position_buckets = None, None
        else:
            (
                main_relative_position_buckets,
                predict_relative_position_buckets,
            ) = self.compute_buffered_relative_buckets(position_ids)
        predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)

        # add position embeddings
        hidden_states = inputs_embeds + main_stream_pos_embed
        hidden_states = hidden_states.transpose(0, 1)

        ngram_embeddings = self.ngram_embeddings.weight

        # prepare attention mask
        if past_key_values is not None:
            assert (
                hidden_states.size(0) == 1
            ), "At the moment `use_cache` is only supported for `decoder_input_ids` of length 1"

            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, batch_size, 1)
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = None
            extended_predict_attention_mask = None
        else:
            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1)
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = self.prepare_attention_mask(hidden_states, attention_mask)
            extended_predict_attention_mask = self.prepare_predict_attention_mask(hidden_states, attention_mask)

        # prepare encoder attention mask
        if encoder_attention_mask is not None:
            extended_encoder_attention_mask = (
                1.0 - encoder_attention_mask[:, None, :].repeat(self.config.num_attention_heads, 1, 1)
            ) * -10000.0
        else:
            extended_encoder_attention_mask = None

        hidden_states = torch.cat([hidden_states] + ngram_hidden_states, 0)

        if self.embeddings_layer_norm:
            hidden_states = self.embeddings_layer_norm(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # init attentions, hidden_states and cache with empty tuples
        all_main_stream_hidden_states = () if output_hidden_states else None
        all_ngram_stream_hidden_states = () if output_hidden_states and self.config.ngram > 0 else None

        all_main_stream_attns = () if output_attentions else None
        all_ngram_stream_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        present_key_values = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_main_stream_hidden_states += (hidden_states[:sequence_length],)
                if self.config.ngram > 0:
                    all_ngram_stream_hidden_states += (hidden_states[sequence_length:],)

            layer_state = past_key_values[idx] if past_key_values is not None else None
            (
                hidden_states,
                layer_self_attn,
                layer_self_predict_attn_output,
                layer_cross_attn,
                layer_past,
            ) = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=extended_encoder_attention_mask,
                layer_state=layer_state,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
                extended_predict_attention_mask=extended_predict_attention_mask,
                main_relative_position_buckets=main_relative_position_buckets,
                predict_relative_position_buckets=predict_relative_position_buckets,
                position_ids=position_ids,
            )
            if use_cache:
                present_key_values += (layer_past,)

            if output_attentions:
                all_main_stream_attns += (layer_self_attn,)
                all_ngram_stream_attns += (layer_self_predict_attn_output,)
                all_cross_attns += (layer_cross_attn,)

        # split last_hidden_state for return
        last_hidden_state = hidden_states[:sequence_length].transpose(0, 1)
        last_hidden_state_ngram = hidden_states[sequence_length:].transpose(0, 1) if self.config.ngram > 0 else None
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1) if encoder_hidden_states is not None else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    last_hidden_state_ngram,
                    present_key_values,
                    all_main_stream_hidden_states,
                    all_ngram_stream_hidden_states,
                    all_main_stream_attns,
                    all_ngram_stream_attns,
                    all_cross_attns,
                ]
                if v is not None
            )
        return ProphetNetDecoderModelOutput(
            last_hidden_state=last_hidden_state,
            last_hidden_state_ngram=last_hidden_state_ngram,
            past_key_values=present_key_values,
            hidden_states=all_main_stream_hidden_states,
            hidden_states_ngram=all_ngram_stream_hidden_states,
            attentions=all_main_stream_attns,
            ngram_attentions=all_ngram_stream_attns,
            cross_attentions=all_cross_attns,
        )

    def compute_buffered_relative_buckets(self, position_ids):
        batch_size, sequence_length = position_ids.shape

        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # buffer relative buckets
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(
            batch_size, 1, 1
        )
        predict_relative_buckets = torch.cat(
            [
                predict_relative_buckets[:, :sequence_length, :sequence_length],
                predict_relative_buckets[
                    :, :sequence_length, self.max_target_positions : self.max_target_positions + sequence_length
                ],
            ],
            2,
        ).repeat(batch_size, 1, 1)

        return main_relative_buckets, predict_relative_buckets

    def prepare_attention_mask(self, hidden_states, attention_mask):
        seq_length, batch_size = hidden_states.shape[:2]

        # get causal mask
        causal_mask = hidden_states.new(seq_length, seq_length).float().fill_(-float("inf"))
        causal_mask = torch.triu(causal_mask, 1)
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, :, :].expand(
            (batch_size,) + causal_mask.shape
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, :]) * -10000.0
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return extended_attention_mask.repeat(self.config.num_decoder_attention_heads, 1, 1)

    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        seq_length, batch_size = hidden_states.shape[:2]

        # get causal mask
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
        )
        predict_causal_mask = torch.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[:, :seq_length, self.max_target_positions : self.max_target_positions + seq_length],
            ],
            dim=-1,
        )
        extended_predict_causal_mask = predict_causal_mask[:, None, :, :].expand(
            predict_causal_mask.shape[:1] + (batch_size,) + predict_causal_mask.shape[1:]
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[None, :, None, :]) * -10000.0
            extended_attention_mask = extended_attention_mask.expand((self.ngram, batch_size, seq_length, seq_length))
            # predicted stream attention_mask should always be 0
            extended_attention_mask = torch.cat(
                [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        return extended_predict_attention_mask.repeat(1, self.config.num_decoder_attention_heads, 1, 1)


@add_start_docstrings(
    "The bare ProphetNet Model transformer outputting raw hidden-states without any specific head on top.",
    PROPHETNET_START_DOCSTRING,
    PROPHETNET_INPUTS_DOCSTRING,
)
class ProphetNetModel(ProphetNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = ProphetNetEncoder(encoder_config, self.word_embeddings)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = ProphetNetDecoder(decoder_config, self.word_embeddings)

        self.init_weights()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
        self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(PROPHETNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="microsoft/prophetnet-large-uncased")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache == use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        return ProphetNetSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_hidden_state_ngram=decoder_outputs.last_hidden_state_ngram,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_ngram_hidden_states=decoder_outputs.hidden_states_ngram,
            decoder_attentions=decoder_outputs.attentions,
            decoder_ngram_attentions=decoder_outputs.ngram_attentions,
            decoder_cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "The ProphetNet Model with a language modeling head. Can be used for summarization.", PROPHETNET_START_DOCSTRING
)
class ProphetNetForConditionalGeneration(ProphetNetPreTrainedModel):
    def __init__(self, config: ProphetNetConfig):
        super().__init__(config)
        self.prophetnet = ProphetNetModel(config)
        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    @add_start_docstrings_to_callable(PROPHETNET_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        outputs = self.prophetnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sequence_length = (
            decoder_input_ids.shape if decoder_input_ids is not None else decoder_inputs_embeds.shape[:2]
        )

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
        else:
            return ProphetNetSeq2SeqLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_ngram_hidden_states=outputs.decoder_ngram_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                decoder_ngram_attentions=outputs.decoder_ngram_attentions,
                decoder_cross_attentions=outputs.decoder_cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def _compute_loss(self, logits, labels):
        expend_targets = labels.new_zeros(self.config.ngram, labels.size(0), labels.size(1)).fill_(self.padding_idx)

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        lprobs = F.log_softmax(
            logits.view(-1, logits.size(-1)),
            dim=-1,
            dtype=torch.float32,
        )

        loss = F.nll_loss(lprobs, expend_targets.view(-1), reduction="sum")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = expend_targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.config.eps / lprobs.size(-1)
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."

        if past:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        if (cur_len == 1) or (cur_len == max_length - 1 and self.config.eos_token_id is not None):
            return self._force_token_ids_generation(logits, self.config.bos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "Scores should be of rank 2"
        scores[:, all_but_token_ids_mask] = -float("inf")
        return scores

    @staticmethod
    def _reorder_cache(past, beam_idx):
        # this function reorders the cache for beam search
        def _reorder_cache(cache_dict, beam_idx):
            for k, key_value_states in cache_dict.items():
                if key_value_states is not None:
                    cache_dict[k] = key_value_states.index_select(0, beam_idx)
            return cache_dict

        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_cache(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder
