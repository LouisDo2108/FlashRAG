from copy import deepcopy
from typing import List

import numpy as np
from flashrag.generator import BaseGenerator
from flashrag.generator.utils import resolve_max_tokens
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        self.model = None

        # if self.model_path == "":
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],  # Limit batch size
            max_model_len=config["max_model_len"],  # Limit context window
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            generation_config="auto",
            disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
            seed=config["seed"],
            enforce_eager=config["enforce_eager"], # disable graph capturing completely if True
        )
        # else:
        #     self.model = LLM(
        #         self.model_path,
        #         tensor_parallel_size=self.tensor_parallel_size,
        #         gpu_memory_utilization=self.gpu_memory_utilization,
        #         max_logprobs=32016,
        #         max_model_len=self.max_model_len,
        #     )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = (
            None
            if "generator_lora_path" not in self._config
            else self._config["generator_lora_path"]
        )
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config["generator_max_input_len"]

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        **params,
    ):

        if isinstance(input_list, str):
            input_list = [input_list]

        # Applying Qwen's chat tempolate, disabling thinking, according to https://qwen.readthedocs.io/en/latest/deployment/vllm.html#python-library
        messages = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": x}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._config["enable_thinking"],
            )
            for x in input_list
        ]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config["seed"]

        # handle param conflict
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # # fix for llama3
        # if "stop" in generation_params:
        #     generation_params["stop"].append("<|eot_id|>")
        #     generation_params["include_stop_str_in_output"] = True
        # else:
        #     generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(messages, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [output.outputs[0].text for output in outputs]
            base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append(
                    [
                        np.exp(list(score_dict.values())[0].logprob)
                        for score_dict in logprobs
                    ]
                )
            return base_output, scores
        else:
            return base_output

