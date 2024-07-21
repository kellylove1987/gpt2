# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import math
from typing import Optional, Tuple

from .configuration_utils import PretrainedConfig
from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


ROPE_CONFIG_DOCSTRING = r"""
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. IMPORTANT: RoPE scaling expects
        `max_position_embeddings` to remain unchagned -- some methods, like 'longrope', require the original value to
        determine which scaling to apply.
        Expected contents:
            `type` (`str`):
                The scaling strategy to use. Can be one of ['linear', 'dynamic', 'yarn', 'longrope'].
            `factor` (`float`, *optional*):
                Used with all scaling types. The scaling factor to apply to the RoPE embeddings. In most scaling types,
                a `factor` of x will enable the model to handle sequences of length x * `max_position_embeddings`.
            `attention_factor` (`float`, *optional*):
                Optional, used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                computation. If unspecified, it defaults to value recommended by the implementation, using the `factor`
                field to infer the suggested value.
            `beta_fast` (`float`, *optional*):
                Optional, only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                ramp function. If unspecified, it defaults to 32.
            `beta_slow` (`float`, *optional*):
                Optional, only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                ramp function. If unspecified, it defaults to 1.
            `short_factor` (`List[float]`, *optional*):
                Optional, only used with 'longrope'. The scaling factor to be applied to short contexts (<
                `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                size divided by the number of attention heads divided by 2
            `long_factor` (`List[float]`, *optional*):
                Optional, only used with 'longrope'. The scaling factor to be applied to short contexts (<
                `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                size divided by the number of attention heads divided by 2
"""


def _compute_default_rope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    if hasattr(config, "rope_dim"):  # TODO (joao): BC -- remove `if` in v4.45, keep `else`
        dim = config.rope_dim
    else:
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)

    # Then applies linear scaling to the frequencies.
    # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
    # applying scaling to the inverse frequencies is equivalent.
    factor = config.rope_scaling["factor"]
    inv_freq /= factor
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    if hasattr(config, "rope_dim"):  # TODO (joao): BC -- remove `if` in v4.45, keep `else`
        dim = config.rope_dim
    else:
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]
    attention_factor = 1.0  # Unused in this type of RoPE

    # seq_len: default to max_position_embeddings, e.g. at init time
    seq_len = seq_len if seq_len is not None else max_position_embeddings

    # Compute the inverse frequencies
    base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://arxiv.org/abs/2309.00071)

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    # Sets the attention factor as suggested in the paper
    attention_factor = config.rope_scaling.get("attention_factor")
    if attention_factor is None:
        attention_factor = 0.1 * math.log(factor) + 1.0

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast") or 32
    beta_slow = config.rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
        """Find dimension range bounds based on rotations"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, max_position_embeddings)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_mask = 1 - linear_ramp_mask(low, high, dim // 2).float().to(device)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: PretrainedConfig, device: "torch.device", seq_len: Optional[int] = None
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    long_factor = config.rope_scaling["long_factor"]
    short_factor = config.rope_scaling["short_factor"]
    factor = config.rope_scaling.get("factor")
    attention_factor = config.rope_scaling.get("attention_factor")

    # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if hasattr(config, "original_max_position_embeddings"):
        max_position_embeddings = config.original_max_position_embeddings
        expanded_max_position_embeddings = config.max_position_embeddings
        factor = expanded_max_position_embeddings / max_position_embeddings
    else:
        max_position_embeddings = config.max_position_embeddings
        expanded_max_position_embeddings = max_position_embeddings * factor

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(factor) / math.log(max_position_embeddings))

    # Compute the inverse frequencies -- scaled based on the target sequence length
    if expanded_max_position_embeddings > max_position_embeddings:
        ext_factors = torch.tensor(long_factor, dtype=torch.float32, device=device)
    else:
        ext_factors = torch.tensor(short_factor, dtype=torch.float32, device=device)
    inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
    inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

    return inv_freq, attention_factor


# This maps the "type" string field in rope config to the corresponding function to compute the RoPE parameters from
# the model config. You can append new {'type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
}


def _validate_default_rope_parameters(config: PretrainedConfig):
    rope_scaling = config.rope_scaling
    required_keys = {"type"}
    received_keys = set(rope_scaling.keys())
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise ValueError(f"Missing required keys in `rope_scaling`: {missing_keys}")

    unused_keys = received_keys - received_keys
    rope_type = rope_scaling["type"]
    if unused_keys:
        raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'='{rope_type}': {unused_keys}")


def _validate_linear_scaling_rope_parameters(config: PretrainedConfig):
    rope_scaling = config.rope_scaling
    required_keys = {"type", "factor"}
    received_keys = set(rope_scaling.keys())
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise ValueError(f"Missing required keys in `rope_scaling`: {missing_keys}")

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    unused_keys = received_keys - received_keys
    rope_type = rope_scaling["type"]
    if unused_keys:
        raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'='{rope_type}': {unused_keys}")


def _validate_yarn_parameters(config: PretrainedConfig):
    rope_scaling = config.rope_scaling
    required_keys = {"type", "factor"}
    received_keys = set(rope_scaling.keys())
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise ValueError(f"Missing required keys in `rope_scaling`: {missing_keys}")

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    optional_keys = {"attention_factor", "beta_fast", "beta_slow"}
    unused_keys = received_keys - required_keys - optional_keys
    rope_type = rope_scaling["type"]
    if unused_keys:
        raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'={rope_type}: {unused_keys}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor < 0:
        raise ValueError(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        raise ValueError(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        raise ValueError(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        raise ValueError(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )


def _validate_longrope_parameters(config: PretrainedConfig):
    rope_scaling = config.rope_scaling
    required_keys = {"type", "short_factor", "long_factor"}
    received_keys = set(rope_scaling.keys())
    missing_keys = required_keys - received_keys
    if missing_keys:
        raise ValueError(f"Missing required keys in `rope_scaling`: {missing_keys}")

    optional_keys = {"attention_factor", "factor"}
    unused_keys = received_keys - required_keys - optional_keys
    rope_type = rope_scaling["type"]
    if unused_keys:
        raise ValueError(f"Unrecognized keys in `rope_scaling` for 'type'={rope_type}: {unused_keys}")

    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)

    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        raise ValueError(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if not len(short_factor) == dim // 2:
        raise ValueError(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        raise ValueError(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if not len(long_factor) == dim // 2:
        raise ValueError(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        logger.warning_once(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_scaling.get("factor")
        if factor is None:
            raise ValueError("Missing required keys in `rope_scaling`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor < 0:
            raise ValueError(
                f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_linear_scaling_rope_parameters,  # `dynamic` has the same validation pattern as `linear`
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
}


def rope_config_validation(config: PretrainedConfig):
    """
    Validate the RoPE config arguments, given a `PretrainedConfig` object
    """
    rope_scaling = config.rope_scaling
    if rope_scaling is None:
        return

    possible_rope_types = set(ROPE_INIT_FUNCTIONS.keys())
    rope_type = rope_scaling.get("type")
    if rope_type is None:
        raise ValueError(
            f"rope_scaling must contain a non-None 'type' field. Possible options are {possible_rope_types}"
        )

    validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
    if validation_fn is not None:
        validation_fn(config)
    # else: no validation, it is a registered custom RoPE type without validation