# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert YOLOS checkpoints from the original repository. URL: https://github.com/hustvl/YOLOS"""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import DetrFeatureExtractor, YolosConfig, YolosForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_yolos_config(yolos_name):
    config = YolosConfig()

    # size of the architecture
    if "yolos_ti" in yolos_name:
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
        config.image_size = [800, 1333]
        config.use_mid_position_embeddings = False
    elif yolos_name == "yolos_s_dWr":
        config.hidden_size = 330
        config.num_hidden_layers = 14
        config.num_attention_heads = 6
        config.intermediate_size = 1320
    elif "yolos_s" in yolos_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif "yolos_b" in yolos_name:
        config.image_size = [800, 1344]

    config.num_labels = 91
    repo_id = "datasets/huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def rename_key(name):
    if "backbone" in name:
        name = name.replace("backbone", "yolos")
    if "cls_token" in name:
        name = name.replace("cls_token", "embeddings.cls_token")
    if "det_token" in name:
        name = name.replace("det_token", "embeddings.det_token")
    if "mid_pos_embed" in name:
        name = name.replace("mid_pos_embed", "encoder.mid_position_embeddings")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "embeddings.position_embeddings")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "blocks" in name:
        name = name.replace("blocks", "encoder.layer")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "class_embed" in name:
        name = name.replace("class_embed", "class_labels_classifier")
    if "bbox_embed" in name:
        name = name.replace("bbox_embed", "bbox_predictor")
    if "yolos.norm" in name:
        name = name.replace("yolos.norm", "yolos.layernorm")

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            dim = model.yolos.encoder.layer[layer_num].attention.attention.all_head_size
            if "weight" in key:
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.query.bias"] = val[:dim]
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                orig_state_dict[f"yolos.encoder.layer.{layer_num}.attention.attention.value.bias"] = val[-dim:]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_yolos_checkpoint(yolos_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    """
    config = get_yolos_config(yolos_name)

    # load original state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # load 🤗 model
    model = YolosForObjectDetection(config)
    model.eval()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # Check outputs on an image, prepared by DetrFeatureExtractor
    feature_extractor = DetrFeatureExtractor(format="coco_detection")
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes

    print("Actual logits:", logits[0, :3, :3])
    print("Actual boxes:", pred_boxes[0, :3, :3])

    expected_slice_logits, expected_slice_boxes = None, None
    if yolos_name == "yolos_ti":
        expected_slice_logits = torch.tensor(
            [[-43.3948, -14.6000, -19.4968], [-27.2291, -7.9266, -10.7180], [-44.8437, -22.2187, -34.0474]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.4408, 0.0750, 0.8582], [0.5009, 0.4857, 1.0000], [0.1724, 0.2022, 0.2394]]
        )
    elif yolos_name == "yolos_s_200_pre":
        expected_slice_logits = torch.tensor(
            [[-24.0248, -10.3024, -14.8290], [-42.0392, -16.8200, -27.4334], [-27.2743, -11.8154, -18.7148]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]]
        )
    elif yolos_name == "yolos_s_300_pre":
        expected_slice_logits = torch.tensor()
        expected_slice_boxes = torch.tensor()
    elif yolos_name == "yolos_s_dWr":
        expected_slice_logits = torch.tensor(
            [[-42.8668, -24.1049, -41.1690], [-34.7456, -14.1274, -24.9194], [-33.7898, -12.1946, -25.6495]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5587, 0.2773, 0.0605], [0.5004, 0.3014, 0.9994], [0.4999, 0.1548, 0.9994]]
        )
    elif yolos_name == "yolos_base":
        expected_slice_logits = torch.tensor(
            [[-40.6064, -24.3084, -32.6447], [-55.1990, -30.7719, -35.5877], [-51.4311, -33.3507, -35.6462]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.5555, 0.2794, 0.0655], [0.9049, 0.2664, 0.1894], [0.9183, 0.1984, 0.1635]]
        )
    else:
        raise ValueError(f"Unknown yolos_name: {yolos_name}")

    assert torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {yolos_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub("nielsr/yolos-s")
        model.push_to_hub("nielsr/yolos-s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--yolos_name",
        default="yolos_s_200_pre",
        type=str,
        help="Name of the YOLOS model you'd like to convert. Should be one of 'yolos_ti', 'yolos_s_200_pre', 'yolos_s_300_pre', 'yolos_s_dWr', 'yolos_base'.",
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the original state dict.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_yolos_checkpoint(args.yolos_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
