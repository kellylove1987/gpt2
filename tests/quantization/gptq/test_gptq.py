# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from transformers.testing_utils import (
    is_torch_available,
    require_accelerate,
    require_auto_gptq,
    require_optimum,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
)


if is_torch_available():
    import torch


class GPTQConfigTest(unittest.TestCase):
    def test_bits(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits="")
            GPTQConfig(bits=1)
        GPTQConfig(bits=2)
        GPTQConfig(bits=4)

    def test_dataset(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, dataset="auto_gpt")
        GPTQConfig(bits=2, dataset="c4")
        GPTQConfig(bits=2, dataset="ptb-new")

    def test_damp_percent(self):
        with self.assertRaises(ValueError):
            GPTQConfig(bits=2, damp_percent=10)
            GPTQConfig(bits=2, damp_percent=-1)
            GPTQConfig(bits=2, damp_percent="0")
        GPTQConfig(bits=2, damp_percent=0.01)

    def test_to_dict(self):
        quantization_config = GPTQConfig(bits=2)
        quantization_config.to_dict()

    def test_from_dict(self):
        dict = {"bits": 2}
        quantization_config = GPTQConfig.from_dict(dict)
        self.assertEqual(dict["bits"], quantization_config.bits)

    @require_optimum
    def test_optimum_config(self):
        from optimum.gptq import GPTQQuantizer

        config = GPTQConfig(bits=2)
        optimum_config = GPTQQuantizer.from_dict(config.to_dict())
        self.assertEqual(optimum_config.bits, config.bits)
        new_config = GPTQConfig.from_dict(optimum_config.to_dict())
        self.assertEqual(optimum_config.bits, new_config.bits)


@slow
@require_optimum
@require_auto_gptq
@require_torch_gpu
class GPTQTest(unittest.TestCase):
    model_name = "bigscience/bloom-560m"

    input_text = "Hello my name is"

    EXPECTED_OUTPUTS = set()
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a professional photographer. I")
    EXPECTED_OUTPUTS.add("Hello my name is John and I am a very good looking man.")

    # this seems a little small considering that we are doing 4bit quant but we have a small model and ww don't quantize the embeddings
    EXPECTED_RELATIVE_DIFFERENCE = 1.664253062

    bits = 4
    group_size = 128
    desc_act = False
    disable_exllama = True

    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ]

    device_map = None

    # called only once for all test in this class
    @classmethod
    def setUpClass(cls):
        """
        Setup quantized model
        """
        cls.model_fp16 = AutoModelForCausalLM.from_pretrained(
            cls.model_name, torch_dtype=torch.float16, device_map=cls.device_map
        )
        cls.mem_fp16 = cls.model_fp16.get_memory_footprint()

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name, use_fast=True)

        quantization_config = GPTQConfig(
            bits=cls.bits,
            dataset=cls.dataset,
            tokenizer=cls.tokenizer,
            group_size=cls.group_size,
            desc_act=cls.desc_act,
            disable_exllama=cls.disable_exllama,
        )

        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name,
            torch_dtype=torch.float16,
            device_map=cls.device_map,
            quantization_config=quantization_config,
        )

    def test_memory_footprint(self):
        r"""
        A simple test to check if the model conversion has been done correctly by checking on the
        memory footprint of the converted model
        """

        mem_quantized = self.quantized_model.get_memory_footprint()

        self.assertAlmostEqual(self.mem_fp16 / mem_quantized, self.EXPECTED_RELATIVE_DIFFERENCE)

    def test_quantized_layers_class(self):
        """
        Simple test to check if the model conversion has been done correctly by checking on
        the class type of the linear layers of the converted models
        """
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=self.desc_act,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=self.disable_exllama,
        )
        self.assertTrue(self.quantized_model.transformer.h[0].mlp.dense_4h_to_h.__class__ == QuantLinear)

    def check_inference_correctness(self, model):
        r"""
        Test the generation quality of the quantized model and see that we are matching the expected output.
        Given that we are operating on small numbers + the testing model is relatively small, we might not get
        the same output across GPUs. So we'll generate few tokens (5-10) and check their output.
        """
        # Check that inference pass works on the model
        encoded_input = self.tokenizer(self.input_text, return_tensors="pt")

        # Check the exactness of the results
        output_sequences = model.generate(input_ids=encoded_input["input_ids"].to(0), max_new_tokens=10)

        # Get the generation
        self.assertIn(self.tokenizer.decode(output_sequences[0], skip_special_tokens=True), self.EXPECTED_OUTPUTS)

    def test_generate_quality(self):
        """
        Simple test to check the quality of the model by comapring the the generated tokens with the expected tokens
        """
        if self.device_map is None:
            self.check_inference_correctness(self.quantized_model.to(0))
        else:
            self.check_inference_correctness(self.quantized_model)

    def test_serialization(self):
        """
        Test the serialization of the model and the loading of the quantized weights works
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if self.disable_exllama:
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname).to(0)
            else:
                # we need to put it directly to the gpu. Otherwise, we won't be able to initialize the exllama kernel
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map={"": 0})
            self.check_inference_correctness(quantized_model_from_saved)

    @require_accelerate
    def test_serialization_big_model_inference(self):
        """
        Test the serialization of the model and the loading of the quantized weights with big model inference
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="auto")
            self.check_inference_correctness(quantized_model_from_saved)

    def test_change_loading_attributes(self):
        """
        Test the serialization of the model and the loading of the quantized weights works with another config file
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.quantized_model.save_pretrained(tmpdirname)
            if self.disable_exllama:
                self.assertEqual(self.quantized_model.config.quantization_config.disable_exllama, True)
                # we need to put it directly to the gpu. Otherwise, we won't be able to initialize the exllama kernel
                quantized_model_from_saved = AutoModelForCausalLM.from_pretrained(
                    tmpdirname, quantization_config=GPTQConfig(disable_exllama=False, bits=6), device_map={"": 0}
                )
                self.assertEqual(quantized_model_from_saved.config.quantization_config.disable_exllama, False)
                self.assertEqual(quantized_model_from_saved.config.quantization_config.bits, self.bits)
                self.check_inference_correctness(quantized_model_from_saved)


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMap(GPTQTest):
    device_map = "auto"


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapExllama(GPTQTest):
    device_map = "auto"
    disable_exllama = False


@require_accelerate
@require_torch_multi_gpu
class GPTQTestDeviceMapCPUOffload(GPTQTest):
    device_map = {
        "transformer.word_embeddings": 0,
        "transformer.word_embeddings_layernorm": 0,
        "lm_head": 0,
        "transformer.h.0": 0,
        "transformer.h.1": 0,
        "transformer.h.2": 0,
        "transformer.h.3": 0,
        "transformer.h.4": 0,
        "transformer.h.5": 0,
        "transformer.h.6": 0,
        "transformer.h.7": 0,
        "transformer.h.8": 0,
        "transformer.h.9": 0,
        "transformer.h.10": 0,
        "transformer.h.11": 0,
        "transformer.h.12": 0,
        "transformer.h.13": 0,
        "transformer.h.14": 0,
        "transformer.h.15": 0,
        "transformer.h.16": 0,
        "transformer.h.17": 0,
        "transformer.h.18": "cpu",
        "transformer.h.19": "cpu",
        "transformer.h.20": "cpu",
        "transformer.h.21": "cpu",
        "transformer.h.22": "cpu",
        "transformer.h.23": 0,
        "transformer.ln_f": 0,
    }
