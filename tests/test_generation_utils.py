# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

from transformers import is_torch_available
from transformers.generation_utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput
from transformers.testing_utils import require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import BartForConditionalGeneration, BartTokenizer, top_k_top_p_filtering
    from transformers.generation_beam_search import BeamSearchScorer
    from transformers.generation_logits_process import (
        HammingDiversityLogitsProcessor,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )


class GenerationTesterMixin:
    model_tester = None
    all_generative_model_classes = ()

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # cut to half length & take max batch_size 3
        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = input_ids.shape[-1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            config.pad_token_id = config.eos_token_id
        return config, input_ids, attention_mask, max_length

    @staticmethod
    def _get_logits_processor_and_kwargs(input_length, eos_token_id, diversity_penalty=None):
        process_kwargs = {
            "min_length": input_length + 1,
            "bad_words_ids": [[1, 0]],
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,
        }
        logits_processor = LogitsProcessorList(
            (
                [
                    HammingDiversityLogitsProcessor(diversity_penalty, num_beams=2, num_beam_groups=2),
                ]
                if diversity_penalty is not None
                else []
            )
            + (
                [
                    MinLengthLogitsProcessor(process_kwargs["min_length"], eos_token_id),
                ]
                if eos_token_id is not None
                else []
            )
            + [
                NoBadWordsLogitsProcessor(process_kwargs["bad_words_ids"], eos_token_id),
                NoRepeatNGramLogitsProcessor(process_kwargs["no_repeat_ngram_size"]),
                RepetitionPenaltyLogitsProcessor(process_kwargs["repetition_penalty"]),
            ]
        )
        return process_kwargs, logits_processor

    @staticmethod
    def _get_warper_and_kwargs(num_beams):
        warp_kwargs = {"top_k": 10, "top_p": 0.7, "temperature": 0.7}
        logits_warper = LogitsProcessorList(
            [
                TemperatureLogitsWarper(warp_kwargs["temperature"]),
                TopKLogitsWarper(top_k=warp_kwargs["top_k"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
                TopPLogitsWarper(top_p=warp_kwargs["top_p"], min_tokens_to_keep=(2 if num_beams > 1 else 1)),
            ]
        )
        return warp_kwargs, logits_warper

    @staticmethod
    def _get_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_diverse_beam_scorer_and_kwargs(batch_size, max_length, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=beam_kwargs["num_beams"],
            device=torch_device,
            length_penalty=beam_kwargs["length_penalty"],
            do_early_stopping=beam_kwargs["early_stopping"],
            num_beam_hyps_to_keep=num_return_sequences,
            num_beam_groups=beam_kwargs["num_beam_groups"],
        )
        return beam_kwargs, beam_scorer

    @staticmethod
    def _get_encoder_outputs(
        model, input_ids, attention_mask, output_attentions=None, output_hidden_states=None, num_interleave=1
    ):
        encoder = model.get_encoder()
        encoder_outputs = encoder(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
            num_interleave, dim=0
        )
        input_ids = torch.zeros_like(input_ids[:, :1]) + model._get_decoder_start_token_id()
        attention_mask = None
        return encoder_outputs, input_ids, attention_mask

    def _greedy_generate(
        self,
        model,
        input_ids,
        attention_mask,
        max_length,
        output_scores=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    ):

        logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
            input_ids.shape[-1], model.config.eos_token_id
        )

        kwargs = {}
        if model.config.is_encoder_decoder:
            max_length = 4

        output_generate = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_length=max_length,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **logits_process_kwargs,
        )

        if model.config.is_encoder_decoder:
            encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                model,
                input_ids,
                attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            kwargs["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            output_greedy = model.greedy_search(
                input_ids,
                max_length=max_length,
                attention_mask=attention_mask,
                logits_processor=logits_processor,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return output_greedy, output_generate

    def test_greedy_generate(self):
        # check `generate()` and `greedy_search()` are equal
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model, input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
            )
            self.assertListEqual(output_greedy.tolist(), output_generate.tolist())

    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
            # disable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            config.use_cache = False
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            if model.config.is_encoder_decoder:
                self.assertIsInstance(output_greedy, GreedySearchEncoderDecoderOutput)
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertIsInstance(output_greedy, GreedySearchDecoderOnlyOutput)
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outptus(output, input_ids, model.config)

    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            # enable cache
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            if not hasattr(config, "use_cache"):
                # only relevant if model has "use_cache"
                return

            config.use_cache = True
            model = model_class(config).to(torch_device).eval()
            output_greedy, output_generate = self._greedy_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

            self.assertListEqual(output_generate.sequences.tolist(), output_greedy.sequences.tolist())

            for output in (output_greedy, output_generate):
                self._check_outptus(output, input_ids, model.config, use_cache=True)

    def _check_outptus(self, output, input_ids, config, use_cache=False):
        batch_size, seq_length = input_ids.shape
        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # Logits
        self._check_logits(batch_size, output.logits, length=gen_len, config=config)

        # Attentions
        if config.is_encoder_decoder:
            # encoder
            encoder_expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
            self.assertIsInstance(output.encoder_attentions, tuple)
            self.assertTrue(
                all(layer_attentions.shape == encoder_expected_shape for layer_attentions in output.encoder_attentions)
            )
            # decoder
            self._check_attentions_for_generate(
                batch_size,
                output.decoder_attentions,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            attentions = output.attentions if not use_cache else output.attentions[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_attentions_for_generate(
                batch_size,
                attentions=attentions,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            encoder_expected_shape = (batch_size, seq_length, config.hidden_size)
            self.assertIsInstance(output.encoder_hidden_states, tuple)
            self.assertTrue(
                all(
                    layer_attentions.shape == encoder_expected_shape
                    for layer_attentions in output.encoder_hidden_states
                )
            )
            # decoder
            self._check_hidden_states_for_generate(
                batch_size,
                output.decoder_hidden_states,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            hidden_states = output.hidden_states if not use_cache else output.hidden_states[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_hidden_states_for_generate(
                batch_size,
                hidden_states,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

    def _check_logits(self, batch_size, logits, length, config, num_beams=None):
        expected_shape = (
            (batch_size, config.vocab_size) if num_beams is None else (batch_size, num_beams, config.vocab_size)
        )
        self.assertIsInstance(logits, tuple)
        self.assertTrue(len(logits) == length)
        self.assertTrue(all(iter_logits.shape == expected_shape for iter_logits in logits))

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beams=None
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertTrue(all(isinstance(iter_attentions, tuple) for iter_attentions in attentions))
        self.assertTrue(len(attentions) == (max_length - min_length))

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            src_len = min_length + idx

            expected_shape = (
                (batch_size, config.num_attention_heads, tgt_len, src_len)
                if num_beams is None
                else (batch_size, num_beams, config.num_attention_heads, tgt_len, src_len)
            )
            # check attn size
            self.assertTrue(all(layer_attention.shape == expected_shape for layer_attention in iter_attentions))

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beams=None
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertTrue(all(isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states))
        self.assertTrue(len(hidden_states) == (max_length - min_length))

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            expected_shape = (batch_size, seq_len, config.hidden_size)
            expected_shape = (
                (batch_size, seq_len, config.hidden_size)
                if num_beams is None
                else (batch_size, num_beams, seq_len, config.hidden_size)
            )
            # check hidden size
            self.assertTrue(
                all(layer_hidden_states.shape == expected_shape for layer_hidden_states in iter_hidden_states)
            )

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id
            )
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `sample()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4

            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                do_sample=True,
                num_beams=1,
                max_length=max_length,
                attention_mask=attention_mask,
                **logits_warper_kwargs,
                **process_kwargs,
            )

            torch.manual_seed(0)
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask
                )
                kwargs["encoder_outputs"] = encoder_outputs
            else:
                attention_mask_clone = attention_mask
                input_ids_clone = input_ids

            with torch.no_grad():
                output_ids_sample = model.sample(
                    input_ids_clone,
                    attention_mask=attention_mask_clone,
                    max_length=max_length,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_sample.tolist())

            # check `generate()` and `sample()` yield equal results for `num_return_sequences`
            num_return_sequences = 3
            if model.config.is_encoder_decoder:
                max_length = 4

            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                do_sample=True,
                num_beams=1,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                attention_mask=attention_mask,
                **logits_warper_kwargs,
                **process_kwargs,
            )

            torch.manual_seed(0)
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=num_return_sequences
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(num_return_sequences, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(num_return_sequences, dim=0)
                input_ids_clone = input_ids.repeat_interleave(num_return_sequences, dim=0)

            with torch.no_grad():
                output_ids_sample = model.sample(
                    input_ids_clone,
                    attention_mask=attention_mask_clone,
                    max_length=max_length,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_sample.tolist())

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id
            )

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `beam_search()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )

            # beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_beam_search = model.beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_search.tolist())

            # check `generate()` and `beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )

            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )
            # beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_beam_search = model.beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_search.tolist())

    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()
            print("Return dict", config.return_dict)
            logits_warper_kwargs, logits_warper = self._get_warper_and_kwargs(num_beams=1)

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `beam_search()` are equal
            # change `num_return_sequences = 2` but not for `beam_scorer`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_beam_scorer_and_kwargs(
                input_ids.shape[0] * num_return_sequences, max_length
            )
            beam_kwargs["num_return_sequences"] = num_return_sequences
            torch.manual_seed(0)
            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_length=max_length,
                **beam_kwargs,
                **logits_warper_kwargs,
            )
            # beam_search does not automatically interleave `batch_size` dim for `num_beams * num_return_sequences`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids, attention_mask = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams * num_return_sequences
                )
                kwargs["encoder_outputs"] = encoder_outputs
            else:
                attention_mask = attention_mask.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0)

            torch.manual_seed(0)
            with torch.no_grad():
                output_ids_beam_sample = model.beam_sample(
                    input_ids.repeat_interleave(beam_scorer.num_beams * num_return_sequences, dim=0),
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask,
                    logits_warper=logits_warper,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_sample.tolist())

        def test_generate_without_input_ids(self):
            config, _, _, max_length = self._get_input_ids_and_config()

            # if no bos token id => cannot generate from None
            if config.bos_token_id is None:
                return

            for model_class in self.all_generative_model_classes:
                model = model_class(config).to(torch_device)
                model.eval()

                output_ids_generate = model.generate(
                    do_sample=False,
                    max_length=max_length,
                )

                self.assertIsNotNone(output_ids_generate)

    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, input_ids, attention_mask, max_length = self._get_input_ids_and_config()

            logits_process_kwargs, logits_processor = self._get_logits_processor_and_kwargs(
                input_ids.shape[-1], config.eos_token_id, diversity_penalty=2.0
            )

            model = model_class(config).to(torch_device)
            model.eval()

            # check `generate()` and `group_beam_search()` are equal
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(input_ids.shape[0], max_length)
            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )

            # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_group_beam_search = model.group_beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_group_beam_search.tolist())

            # check `generate()` and `group_beam_search()` are equal for `num_return_sequences`
            num_return_sequences = 2
            if model.config.is_encoder_decoder:
                max_length = 4
            beam_kwargs, beam_scorer = self._get_diverse_beam_scorer_and_kwargs(
                input_ids.shape[0], max_length, num_return_sequences=num_return_sequences
            )

            output_ids_generate = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=max_length,
                **beam_kwargs,
                **logits_process_kwargs,
            )
            # group_beam_search does not automatically interleave `batch_size` dim for `num_beams`
            kwargs = {}
            if model.config.is_encoder_decoder:
                encoder_outputs, input_ids_clone, attention_mask_clone = self._get_encoder_outputs(
                    model, input_ids, attention_mask, num_interleave=beam_scorer.num_beams
                )
                kwargs["encoder_outputs"] = encoder_outputs
                input_ids_clone = input_ids_clone.repeat_interleave(beam_scorer.num_beams, dim=0)
            else:
                attention_mask_clone = attention_mask.repeat_interleave(beam_scorer.num_beams, dim=0)
                input_ids_clone = input_ids.repeat_interleave(beam_scorer.num_beams, dim=0)

            with torch.no_grad():
                output_ids_beam_search = model.group_beam_search(
                    input_ids_clone,
                    beam_scorer,
                    max_length=max_length,
                    attention_mask=attention_mask_clone,
                    logits_processor=logits_processor,
                    **kwargs,
                )
            self.assertListEqual(output_ids_generate.tolist(), output_ids_beam_search.tolist())


@require_torch
class UtilsFunctionsTest(unittest.TestCase):

    # tests whether the top_k_top_p function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = torch.tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 4 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 4 highest values <= 0.6
            ],
            dtype=torch.float,
            device=torch_device,
        )

        non_inf_expected_idx = torch.tensor(
            [[0, 0], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 20], [1, 27]],
            dtype=torch.long,
            device=torch_device,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = torch.tensor(
            [
                8.2221,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype=torch.float,
            device=torch_device,
        )

        output = top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)
        non_inf_output = output[output != -float("inf")].to(device=torch_device)
        non_inf_idx = (output != -float("inf")).nonzero().to(device=torch_device)

        self.assertTrue(torch.allclose(non_inf_expected_output, non_inf_output, atol=1e-12))
        self.assertTrue(torch.all(torch.eq(non_inf_expected_idx, non_inf_idx)))


@require_torch
class GenerationIntegrationTests(unittest.TestCase):
    @slow
    def test_diverse_beam_search(self):
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood.
        The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
        "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
        The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both."""

        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = bart_model.generate(
            input_ids, num_beams=4, num_return_sequences=2, num_beam_groups=4, diversity_penalty=2.0
        )

        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The couple announced the birth of their son, Silas Randall Timberlake, in a statement. Silas was the middle name of Timberlake's maternal grandfather Bill Bomar. Randall is the musician's own middle name, as well as his father's first. It is the first baby for both of them.",
                "Justin Timberlake and Jessica Biel have a son. The baby is named Silas Randall Timberlake. It is the first child for both. The couple announced the pregnancy in January. The name Silas is the middle name of Timberlake's maternal grandfather. It's also his own middle name.",
            ],
        )
