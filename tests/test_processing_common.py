# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


import inspect
import json
import tempfile


try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import CLIPTokenizerFast, ProcessorMixin
from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_tokenizers,
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import CLIPImageProcessor


@require_torch
@require_vision
@require_torch
class ProcessorTesterMixin:
    processor_class = None
    text_data_arg_name = "input_ids"
    images_data_arg_name = "pixel_values"
    videos_data_arg_name = "pixel_values_videos"

    def prepare_processor_dict(self):
        return {}

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            component_class_name = component_class_name[0]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if attribute == "tokenizer" and not component.pad_token:
            component.pad_token = "[TEST_PAD]"

        return component

    def prepare_components(self):
        components = {}
        for attribute in self.processor_class.attributes:
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        components = self.prepare_components()
        processor = self.processor_class(**components, **self.prepare_processor_dict())
        return processor

    @require_vision
    def prepare_image_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
        return image_inputs

    @require_vision
    def prepare_video_inputs(self):
        video_file = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
        )
        return [np.load(video_file)]

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            self.assertEqual(obj[key], value)
            self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            if len(saved_files) > 0:
                check_json_file_has_correct_format(saved_files[0])
                processor_second = self.processor_class.from_pretrained(tmpdirname)

                self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

    # These kwargs-related tests ensure that processors are correctly instantiated.
    # they need to be applied only if an image_processor exists.

    def skip_processor_without_typed_kwargs(self, processor):
        # TODO this signature check is to test only uniformized processors.
        # Once all are updated, remove it.
        is_kwargs_typed_dict = False
        call_signature = inspect.signature(processor.__call__)
        for param in call_signature.parameters.values():
            if param.kind == param.VAR_KEYWORD and param.annotation != param.empty:
                is_kwargs_typed_dict = (
                    hasattr(param.annotation, "__origin__") and param.annotation.__origin__ == Unpack
                )
        if not is_kwargs_typed_dict:
            self.skipTest(f"{self.processor_class} doesn't have typed kwargs.")

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 117)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", size=(234, 234), crop_size=(234, 234)
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 234)

    @require_torch
    @require_vision
    def test_video_processor_defaults_preserved_by_kwargs(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234), crop_size=(234, 234))
        video_processor = self.get_component("video_processor", size=(234, 234), crop_size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_input, return_tensors="pt")
        self.assertEqual(inputs[self.videos_data_arg_name].shape[-1], 234)

    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 112)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component("image_processor", size=(234, 234))
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, size=[224, 224], crop_size=(224, 224), return_tensors="pt"
        )
        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 224)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            crop_size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 214)
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            crop_size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 214)
        self.assertEqual(len(inputs[self.text_data_arg_name][0]), len(inputs[self.text_data_arg_name][1]))

    @require_torch
    @require_vision
    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"size": {"height": 222, "width": 222}},
                size={"height": 214, "width": 214},
                crop_size={"height": 214, "width": 214},
                return_tensors="pt",
            )

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {
                "size": {"height": 214, "width": 214},
                "crop_size": {"height": 214, "width": 214},
            },
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 214)
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {
                "size": {"height": 214, "width": 214},
                "crop_size": {"height": 214, "width": 214},
            },
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], 214)
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)


class MyProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, processor_attr_1=1, processor_attr_2=True):
        super().__init__(image_processor, tokenizer)

        self.processor_attr_1 = processor_attr_1
        self.processor_attr_2 = processor_attr_2


@require_tokenizers
@require_vision
class ProcessorTest(unittest.TestCase):
    processor_class = MyProcessor

    def prepare_processor_dict(self):
        return {"processor_attr_1": 1, "processor_attr_2": False}

    def get_processor(self):
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")
        processor = MyProcessor(image_processor, tokenizer, **self.prepare_processor_dict())

        return processor

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            self.assertEqual(obj[key], value)
            self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            processor_second = self.processor_class.from_pretrained(tmpdirname)

        self.assertEqual(processor_second.to_dict(), processor_first.to_dict())
