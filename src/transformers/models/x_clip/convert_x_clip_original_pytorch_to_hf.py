# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import json

import numpy as np
import torch
from PIL import Image

import requests
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, XClipConfig, XClipModel


def get_xclip_config(model_name):
    config = XClipConfig()
    return config


def rename_key(name):
    # text encoder
    if name == "token_embedding.weight":
        name = name.replace("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    if name == "positional_embedding":
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if name.startswith("transformer.resblocks"):
        name = name.replace("transformer.resblocks", "text_model.encoder.layers")
    if "attn.out_proj" in name:
        name = name.replace("attn.out_proj", "self_attn.out_proj")
    # visual encoder

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        dim = config.text_config.hidden_size
        
        print("Old key:", key)
        
        if "attn.in_proj" in key and "visual" not in key:
            key_split = key.split(".")
            layer_num = key_split[2]
            if "weight" in key:
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"text_model.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_key_name = rename_key(key)
            print("New key:", new_key_name)
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_xclip_checkpoint(checkpoint_url, model_name, pytorch_dump_folder_path):
    config = get_xclip_config(model_name)
    model = XClipModel(config)
    model.eval()

    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)['model']
    state_dict = convert_state_dict(state_dict, config)
    
    model = XClipModel(config)
    model.load_state_dict(state_dict)

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # timm_outs = timm_model(inputs["pixel_values"])
    # hf_outs = model(**inputs).logits

    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    # model.save_pretrained(pytorch_dump_folder_path)

    # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_8.pth",
        type=str,
        help="URL fo the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--model_name",
        default="xclip-base-patch32",
        type=str,
        help="Name of the model.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_xclip_checkpoint(args.checkpoint_url, args.model_name, args.pytorch_dump_folder_path)