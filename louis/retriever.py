import json
import os

import warnings
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel, LoraConfig

from tevatron.retriever.modeling import EncoderModel
from flashrag.retriever import BaseTextRetriever, load_corpus, load_docs
from copy import deepcopy


os.environ["TOKENIZERS_PARALLELISM"] = "false"
_has_printed_instruction = False


class Encoder(EncoderModel):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 instruction: str = "",
                 data_args=None,
                 ):
        super().__init__(encoder, pooling, normalize, temperature)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.data_args = data_args

    def encode_query(self, qry):
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":
            query_hidden_states = self.encoder(**qry, return_dict=True)
            query_hidden_states = query_hidden_states.last_hidden_state
            return self._pooling(query_hidden_states, qry["attention_mask"])
        else:
            task = 'retrieval.query'
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full((qry['input_ids'].size(0),), task_id, dtype=torch.int32, device=qry['input_ids'].device)
            query_hidden_states = self.encoder(
                **qry, return_dict=True, adapter_mask=adapter_mask,
            )
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]
            return self._pooling(query_hidden_states, qry["attention_mask"])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":
            return self.encode_query(psg)
        else:
            task = "retrieval.passage"
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full((psg["input_ids"].size(0),),task_id,dtype=torch.int32,device=psg["input_ids"].device,)
            query_hidden_states = self.encoder(**psg, return_dict=True, adapter_mask=adapter_mask,)
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]
            return self._pooling(query_hidden_states, psg["attention_mask"])

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def parse_query(self, query_list):
        """
        processing query for different encoders
        """
        global _has_printed_instruction

        if isinstance(query_list, str):
            query_list = [query_list]

        if self.instruction is not None:
            self.instruction = self.instruction.strip() + " " if self.instruction else ""
        else:
            self.instruction = self.set_default_instruction(is_zh=judge_zh(query_list[0]))

        if not _has_printed_instruction:
            if self.instruction == "":
                warnings.warn('Instruction is not set')
            else:
                print(f"Use `{self.instruction}` as retrieval instruction")
            _has_printed_instruction = True

        query_list = [self.instruction + query for query in query_list]

        return query_list

    def set_default_instruction(self, is_zh=False): 
        model_name = self.model_args.model_name_or_path.lower()
        instruction = ""  

        if "e5" in model_name:
            if self.data_args.encode_is_query:
                instruction = "query: "
            else:
                instruction = "passage: "

        if "bge" in model_name:
            if self.data_args.encode_is_query:
                if "zh" in model_name.lower() or is_zh:
                    instruction = "为这个句子生成表示以用于检索相关文章："
                else:
                    instruction = (
                        "Represent this sentence for searching relevant passages: "
                    )


        return instruction

    @torch.inference_mode()
    def single_batch_encode(
        self, query_list: Union[List[str], str], is_query=True
    ) -> np.ndarray:
        query_list = self.parse_query(query_list)

        inputs = self.tokenizer(
            query_list,
            padding=True,
            truncation=True,
            max_length=self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len,
            return_tensors="pt",
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        inputs = {k: v.cuda() for k, v in inputs.items()}

        # if "T5" in type(self.model).__name__ or (
        #     isinstance(self.model, torch.nn.DataParallel)
        #     and "T5" in type(self.model.module).__name__
        # ):
        #     # T5-based retrieval model
        #     decoder_input_ids = torch.zeros(
        #         (inputs["input_ids"].shape[0], 1), dtype=torch.long
        #     ).to(inputs["input_ids"].device)
        #     output = self.model(
        #         **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
        #     )
        #     query_emb = output.last_hidden_state[:, 0, :]

        # else:
        # output = self.model(**inputs, return_dict=True)
        # pooler_output = output.get("pooler_output", None)
        # last_hidden_state = output.get("last_hidden_state", None)
        # query_emb = pooling(
        #     pooler_output,
        #     last_hidden_state,
        #     inputs["attention_mask"],
        #     self.pooling_method,
        # )
        # if "dpr" not in self.model_name:
        #     query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        if is_query:
            emb = self.encode_query(inputs)
        else:
            emb = self.encode_passage(inputs)

        emb = emb.cpu().detach().numpy() # Cast from bf16/fp16 to fp32 .float()
        # emb = emb.astype(np.float32, order="C")
        return emb

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=True, dtype=torch.float32) -> np.ndarray:
        query_emb = []
        with (torch.autocast("cuda", dtype=dtype)):
            with torch.no_grad():
                for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: "):
                    query_emb.append(
                        self.single_batch_encode(query_list[i : i + batch_size], is_query)
                    )
        # query_emb = np.concatenate(query_emb, axis=0)
        query_emb = np.concatenate(query_emb)
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(
        self, query_list: Union[List[str], str], batch_size=64, is_query=True
    ) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode_query(query_list, batch_size, is_query)
        return query_emb

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             instruction: str = "",
             data_args=None,
             tokenizer: AutoTokenizer = None,
             **hf_kwargs):

        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)

            """
            Slightly modify how to load the LoRA fine-tuned model to disable this warning: 
            UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
            This is because the base_model will load a checkpoint that already has peft_config attribute in it.
            """
            # base_model = cls.TRANSFORMER_CLS.from_pretrained(
            #     lora_config.base_model_name_or_path, # type: ignore
            #     weights_only=False,
            #     trust_remote_code=True,
            #     **hf_kwargs,
            # )
            try:
                _hf_kwargs = deepcopy(hf_kwargs)
                _hf_kwargs.pop("attn_implementation")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    lora_config.base_model_name_or_path,
                    trust_remote_code=True,
                    weights_only=False,
                    attn_implementation="flash_attention_2",
                    **_hf_kwargs
                )
                print("Using flash attention 2!")
            except Exception as e:
                print(e)
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    lora_config.base_model_name_or_path,
                    weights_only=False,
                    trust_remote_code=True, 
                    **hf_kwargs
                )
                print(f"Fall back to use {hf_kwargs['attn_implementation']}")

            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = 0

            lora_model = PeftModel.from_pretrained(
                base_model,
                lora_name_or_path,
                config=lora_config,
            )
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                tokenizer=tokenizer,
                data_args=data_args,
                instruction=instruction
            )
            print(f"Loaded LoRAs from {lora_name_or_path}")
        else:
            try:
                _hf_kwargs = deepcopy(hf_kwargs)
                _hf_kwargs.pop("attn_implementation")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=True,
                    weights_only=False,
                    attn_implementation="flash_attention_2",
                    **_hf_kwargs
                )
                print("Using flash attention 2!")
            except Exception as e:
                print(e)
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True, 
                    **hf_kwargs
                )
                print(f"Fall back to use {hf_kwargs['attn_implementation']}")

            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = 0

            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                tokenizer=tokenizer,
                data_args=data_args,
                instruction=instruction
            )
            print(f"Loaded model from {model_name_or_path}")
            print("If your model is PEFT-based, please provide lora_name_or_path to load the PEFT model correctly!!!")
        return model


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, training_args, model_args, data_args, config: dict, corpus=None):

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args

        self._config = config
        self.update_base_setting()
        self.update_additional_setting()

        self.load_corpus(corpus)
        self.load_index()

        self.torch_dtype = torch.float32 # Default
        if training_args.bf16:
            self.torch_dtype = torch.bfloat16
        elif training_args.fp16:
            self.torch_dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if data_args.padding_side == "right":
            tokenizer.padding_side = "right"
        else:
            tokenizer.padding_side = "left"

        self.encoder = Encoder.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=self.torch_dtype,
            attn_implementation=model_args.attn_implementation,
            tokenizer=tokenizer,
            data_args=data_args,
            instruction=self.instruction,
        )
        self.encoder = self.encoder.to(training_args.device)
        self.encoder.eval()

    def load_corpus(self, corpus):
        # if corpus is None:
        #     self.corpus = load_corpus(self.corpus_path)
        # else:
        #     self.corpus = corpus

        try:
            corpus = load_dataset(
                self.data_args.dataset_name,
                "corpus",
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
            if "contents" not in corpus.features:
                try:
                    print("No `contents` field found in corpus, using `text` instead.")
                    corpus = corpus.map(lambda x: {"contents": x["text"]})
                except:
                    warnings.warn("No `contents` & `text` field found in corpus.")
        except:
            corpus_path = self.corpus_path
            if corpus_path.endswith(".jsonl"):
                corpus = load_dataset('json', data_files=corpus_path, split="train")
            elif corpus_path.endswith(".parquet"):
                corpus = load_dataset('parquet', data_files=corpus_path, split="train")
                corpus = corpus.cast_column('image', Image())
            else:
                raise NotImplementedError("Corpus format not supported!")
            if 'contents' not in corpus.features:
                try:
                    print("No `contents` field found in corpus, using `text` instead.")
                    corpus = corpus.map(lambda x: {"contents": x["text"]})
                except:
                    warnings.warn("No `contents` & `text` field found in corpus.")

        self.corpus = corpus

    def load_index(self):
        # if self.index_path is None or not os.path.exists(self.index_path):
        #     raise Warning(f"Index file {self.index_path} does not exist!")
        self.index_path = str(Path(self.data_args.encode_output_path).parent / "dense_Flat.index") # Hard coded
        
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = (
            self._config["instruction"]
            if self._config["instruction"]
            else self.data_args.query_instruction
        )

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(
                open(os.path.join(model_path, "1_Pooling/config.json"))
            )
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(
                            f"Pooling method {detect_pooling_method} is not implemented.",
                            UserWarning,
                        )
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if (
            detect_pooling_method is not None
            and detect_pooling_method != pooling_method
        ):
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk
            query_emb = self.encoder.encode(query, dtype=self.torch_dtype)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size

        results = []
        scores = []
        emb = self.encoder.encode(query, batch_size=batch_size, is_query=True, dtype=self.torch_dtype)

        # Can potentially skip this step since tevatron already encodes the queries once.
        # with open(embedding_path, 'rb') as f:
        #     reps, lookup = pickle.load(f)
        #     all_embeddings = np.array(reps).reshape(-1, hidden_size)

        scores, idxs = self.index.search(emb, k=num)
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = sum(idxs, [])
        results = load_docs(self.corpus, flat_idxs)
        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]

        if return_score:
            return results, scores
        else:
            return results

