import logging
import os
import random
import sys
from pdb import set_trace as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import torch

from datasets import load_dataset
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments

from flashrag.evaluator import Evaluator
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_refiner

from transformers.utils.import_utils import is_torch_available
from transformers.hf_argparser import HfArgumentParser
from vllm.distributed import cleanup_dist_env_and_memory

# Local imports
from config import Config
from dataset import Dataset
from retriever import DenseRetriever
from llm import VLLMGenerator
from vllm.distributed import cleanup_dist_env_and_memory

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True):
    # Copy from transformers.trainer_utilss.set_seed with some modifications
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB). From https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
            torch.use_deterministic_algorithms(True)

            # # Enable CUDNN deterministic mode
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


class ToyPipeline(SequentialPipeline):
    def __init__(self, training_args, model_args, data_args, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        self.config = config
        self.device = config["device"]
        self.data_args = data_args
        
        self.retriever = None
        self.evaluator = Evaluator(config)
        
        self.save_retrieval_cache = config["save_retrieval_cache"]
        
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template
        
        if generator is None:
            self.generator = VLLMGenerator(config)
        
        self.retriever = DenseRetriever(
            training_args, model_args, data_args, config
        )

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            for metric_name, score in eval_result.items():
                eval_result[metric_name] = round(score, 3)
            print(
                f"Evaluation of {self.data_args.dataset_name}'s {self.data_args.dataset_config}"
            )
            for k, v in eval_result.items():
                print(f"{k}: {v}")
            return eval_result

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc["contents"] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--flashrag_config_yaml_path", type=str, required=True)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        
        # config_dict = {"data_dir": "dataset/"} # If you want to override something
        flashrag_config_yaml_path = other_args.flashrag_config_yaml_path
        flashrag_config = Config(
            config_file_path=flashrag_config_yaml_path,
            # config_dict=config_dict,
        )
        
        flashrag_config["save_dir"] = training_args.output_dir
        if not flashrag_config.final_config.get('disable_save', False):
            flashrag_config._prepare_dir()
        
        
    set_seed(42, deterministic=True) # Ensuring the seed is set properly

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    data = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        data_files=data_args.dataset_path,
        split=data_args.dataset_split,
        cache_dir=data_args.dataset_cache_dir,
        num_proc=data_args.num_proc,
    )
    if data_args.dataset_number_of_shards > 1:
        data = data.shard(
            num_shards=data_args.dataset_number_of_shards,
            index=data_args.dataset_shard_index,
        )
    test_data = Dataset(
        config=flashrag_config,
        data=data
    )

    # ### For quick debugging
    # temp_data = []
    # for ix, item in enumerate(data):
    #     if ix > 10:
    #         break
    #     temp_data.append(item)
    # test_data = Dataset(
    #     config=flashrag_config, # type: ignore
    #     data=temp_data
    # )
    # ### For quick debugging

    prompt_templete = PromptTemplate(
        flashrag_config,
        system_prompt="Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:",
    )

    pipeline = ToyPipeline(
        training_args,
        model_args,
        data_args,
        config=flashrag_config, 
        prompt_template=prompt_templete
    )

    output_dataset = pipeline.run(test_data, do_eval=True)

    if hasattr(pipeline, "model"):
        del pipeline.model
    cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
