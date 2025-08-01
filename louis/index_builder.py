import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pdb import set_trace as st 

import faiss
import json
import warnings
import numpy as np
from typing import Dict, List
import shutil
import subprocess
import argparse
import datasets
import torch
import argparse
from seismic.seismic import SeismicIndex
from tqdm import tqdm
from flashrag.retriever.utils import load_model, load_corpus, pooling, set_default_instruction, judge_zh
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.hf_argparser import HfArgumentParser

import os
import multiprocessing
import pickle

from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments

cores = str(multiprocessing.cpu_count())
os.environ["RAYON_NUM_THREADS"] = cores


class Index_Builder:
    r"""A tool class used to build an index used in retrieval."""

    def __init__(
            self,
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            fp16,
            bf16,
            n_postings=1000,
            centroid_fraction=0.2,
            min_cluster_size=2,
            summary_energy=0.4,
            batched_indexing=10000,
            corpus_embedded_path=None,
            pooling_method=None,
            instruction=None,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False,
            use_sentence_transformer=False,
            bm25_backend="bm25s",
            index_modal="all",
            nknn=0,
            model_args: ModelArguments=None,
            training_args: TrainingArguments=None,
            data_args: DataArguments=None,
    ):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.corpus_embedded_path = corpus_embedded_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.fp16 = fp16
        self.bf16 = bf16
        self.instruction = instruction
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.use_sentence_transformer = use_sentence_transformer
        self.bm25_backend = bm25_backend
        self.index_modal = index_modal
        self.n_postings = n_postings
        self.centroid_fraction = centroid_fraction
        self.min_cluster_size = min_cluster_size
        self.summary_energy = summary_energy
        self.batched_indexing = batched_indexing
        self.nknn = nknn
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        
        self.torch_dtype = torch.float32 # Default
        if training_args.bf16:
            self.torch_dtype = torch.bfloat16
        elif training_args.fp16:
            self.torch_dtype = torch.float16

        # judge if the retrieval model is clip
        self.is_clip = ("clip" in self.retrieval_method) or (self.model_path is not None and "clip" in self.model_path)
        if not self.is_clip:
            try:
                with open(os.path.join(self.model_path, "config.json")) as f:
                    config = json.load(f)
                model_type = config.get("architectures", [None])[0]
                self.is_clip = "clip" in model_type.lower()
            except:
                pass
        if self.is_clip:
            print("Use clip model!")

        # config pooling method
        if pooling_method is None:
            try:
                # read pooling method from 1_Pooling/config.json
                pooling_config = json.load(open(os.path.join(self.model_path, "1_Pooling/config.json")))
                for k, v in pooling_config.items():
                    if k.startswith("pooling_mode") and v == True:
                        pooling_method = k.split("pooling_mode_")[-1]
                        if pooling_method == "mean_tokens":
                            pooling_method = "mean"
                        elif pooling_method == "cls_token":
                            pooling_method = "cls"
                        else:
                            # raise warning: not implemented pooling method
                            warnings.warn(f"Pooling method {pooling_method} is not implemented.", UserWarning)
                            pooling_method = "mean"
                        break
            except:
                print(f"Pooling method not found in {self.model_path}, use default pooling method (mean).")
                # use default pooling method
                pooling_method = "mean"
        else:
            if pooling_method not in ["mean", "cls", "pooler"]:
                raise ValueError(f"Invalid pooling method {pooling_method}.")
        self.pooling_method = pooling_method

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)

        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""

        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            if self.bm25_backend == "pyserini":
                self.build_bm25_index_pyserini()
            elif self.bm25_backend == "bm25s":
                self.build_bm25_index_bm25s()
            else:
                assert False, "Invalid bm25 backend!"
        elif self.retrieval_method == "splade":
            self.build_seismic_index()
        else:
            self.build_dense_index()

    def build_seismic_index(self):
        """Build Seismic index after saving documents in required JSONL format using batch processing."""

        r"""Full Command:
        # Use "splade" (sparse embedding neural model) as retrieval method to trigger sysmic index costruction.
        python -m flashrag.retriever.index_builder \ # builder
        --retrieval_method splade \ # Model name to trigger seismic index (splade only available)
        --model_path retriever/splade-v3 \ # Local path or repository path are both supported.
        --corpus_embedded_path data/ms_marco/ms_marco_embedded_corpus.jsonl \  # Use cached embedded corpus if corpus is already available ins seismic expected format
        --corpus_path data/ms_marco/ms_marco_corpus.jsonl \ # Corpus path in format {id, contents} jsonl file to be embedded if not already built
        --save_dir indexes/ \ # save index directory
        --fp16 \ # tell to use fp16 for splade model
        --max_length 512 \ # max tokens for each document
        --batch_size 4 \ # batch size for splade model (Suggested between 2 and 24 for 16GB VRAM)
        --n_postings 1000 \ # seismic number of posting lists
        --centroid_fraction 0.2 \ # seismic centroids
        --min_cluster_size 2 \ # seismic min cluster
        --summary_energy 0.4 \ # seismic energy
        --batched_indexing 10000 # seismic batch
        --nknn 32
        """

        if self.pooling_method != 'max':
            print(
                f'Pooling method: {self.pooling_method.upper()} not supported on sparse neural retrieval models. fallback to: MAX.')
        # Load document encoder model (only splade i currently implemented at the moment)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForMaskedLM.from_pretrained(self.model_path)

        """
        Remove these lines to enable both fp16 and bf16 inference

        # # Use half precision
        # if self.fp16:
        #     model = model.half()
        
        
        # # Use more devices if available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     model = torch.nn.DataParallel(model)
        """

        # Load to cuda
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()

        # Create file for embedded corpus
        corpus_name = os.path.splitext(os.path.basename(self.corpus_path))[0]
        output_path = os.path.join(self.save_dir, f"{corpus_name}_{self.retrieval_method}_embedded.jsonl")

        # Init vars
        total_docs = len(self.corpus)
        processed_docs = 0
        start_time = time.time()
        last_update = start_time

        if self.corpus_embedded_path:
            print("Using cached corpus in seismic format.")
            if self.nknn > 0:
                index = SeismicIndex.build(
                    self.corpus_embedded_path,
                    n_postings=self.n_postings,
                    centroid_fraction=self.centroid_fraction,
                    min_cluster_size=self.min_cluster_size,
                    summary_energy=self.summary_energy,
                    batched_indexing=self.batched_indexing,
                    num_threads=int(cores),
                    nknn=self.nknn
                )
            else:
                index = SeismicIndex.build(
                    self.corpus_embedded_path,
                    n_postings=self.n_postings,
                    centroid_fraction=self.centroid_fraction,
                    min_cluster_size=self.min_cluster_size,
                    summary_energy=self.summary_energy,
                    batched_indexing=self.batched_indexing,
                    num_threads=int(cores)
                )
            index.save(os.path.join(self.save_dir, f"{corpus_name}_{self.retrieval_method}"))
            return index

        with open(output_path, 'w') as f_out, ThreadPoolExecutor(max_workers=1) as pool:
            # create progress bar
            progress_bar = tqdm(
                desc="Processing Documents",
                total=total_docs,
                unit="doc",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )

            batch = []
            for doc in self.corpus:
                # Load batch
                batch.append(doc)

                # On batch ready process batch
                if len(batch) >= self.batch_size:
                    self._process_batch(batch, f_out, model, tokenizer, pool)
                    processed_docs += len(batch)
                    batch = []

                    # Progress updates
                    current_time = time.time()
                    if processed_docs % 100 == 0 or (current_time - last_update) > 5.0:
                        elapsed = current_time - start_time
                        docs_per_sec = processed_docs / elapsed
                        remaining = (total_docs - processed_docs) / docs_per_sec

                        progress_bar.set_postfix({
                            'speed': f"{docs_per_sec:.1f} docs/s",
                            'ETA': f"{remaining / 60:.1f} min"
                        })
                        progress_bar.update(100 if processed_docs % 100 == 0 else processed_docs % 100)
                        last_update = current_time

            # Handle final leftover batch
            if batch:
                self._process_batch(batch, f_out, model, tokenizer, pool)
                progress_bar.update(len(batch))

            progress_bar.close()

        if self.nknn > 0:
            index = SeismicIndex.build(
                output_path,
                n_postings=self.n_postings,
                centroid_fraction=self.centroid_fraction,
                min_cluster_size=self.min_cluster_size,
                summary_energy=self.summary_energy,
                batched_indexing=self.batched_indexing,
                num_threads=int(cores),
                nknn=self.nknn
            )
        else:
            # Create index on embedded corpus file and save it
            index = SeismicIndex.build(
                output_path,
                n_postings=self.n_postings,
                centroid_fraction=self.centroid_fraction,
                min_cluster_size=self.min_cluster_size,
                summary_energy=self.summary_energy,
                batched_indexing=self.batched_indexing,
                num_threads=int(cores)
            )

        index.save(os.path.join(self.save_dir, f"{corpus_name}_{self.retrieval_method}"))
        return index

    def _process_batch(self, batch, f_out, model, tokenizer, pool):
        # Get embeddings
        texts = [doc['contents'] for doc in batch]
        vectors = self._get_sparse_embedding(texts, model, tokenizer)

        # create json for the batch
        def save(docs, embs):
            for doc, vector in zip(docs, embs):
                if 'vector' not in doc:
                    doc['vector'] = vector
                f_out.write(json.dumps({
                    "id": str(doc['id']),
                    "contents": doc['contents'],
                    "vector": doc['vector']
                }) + "\n")
        pool.submit(save, batch, vectors)

    def _get_sparse_embedding(self, texts: List[str], model, tokenizer) -> List[Dict[str, float]]:
        """Generate sparse embeddings for a batch of texts."""
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
            add_special_tokens=True
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits  # [batch_size, seq_len, vocab_size]
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            scores = torch.log1p(torch.relu(logits)) * attention_mask

            v_repr = torch.max(scores, dim=1)[0]  # [batch_size, vocab_size]

            # Move to CPU (it seems much faster)
            v_repr = v_repr.cpu()
            nonzero_mask = v_repr > 1e-4

            # Get sparse values and indices in batch
            batch_indices, token_indices = torch.nonzero(nonzero_mask, as_tuple=True)
            token_scores = v_repr[batch_indices, token_indices]

            # Convert once all token IDs to strings (batched)
            unique_token_ids = torch.unique(token_indices)
            token_id_to_token = {
                idx.item(): tok for idx, tok in zip(
                    unique_token_ids, tokenizer.convert_ids_to_tokens(unique_token_ids.tolist())
                )
            }

            # Build final embeddings
            from collections import defaultdict
            embeddings = defaultdict(dict)
            for b_idx, t_idx, score in zip(batch_indices, token_indices, token_scores):
                embeddings[b_idx.item()][token_id_to_token[t_idx.item()]] = round(score.item(), 4)

            # Convert to list for each document
            return [embeddings[i] for i in range(len(texts))]

    @staticmethod
    def get_tokens_and_weights(sparse_embedding, tokenizer):
        token_weight_dict = {}
        for i in range(len(sparse_embedding.indices)):
            token = tokenizer.decode([sparse_embedding.indices[i]])
            weight = sparse_embedding.values[i]
            token_weight_dict[token] = round(weight, 4)

        # Sort the dictionary by weights
        token_weight_dict = dict(sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True))
        return token_weight_dict

    @staticmethod
    def clean_text(self, text: str) -> str:
        """Preprocess the text by removing special characters that seems to cause some problems in seismic index build resulting in a parse error."""
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces
        text = text.strip()  # Remove leading/trailing whitespaces
        return text

    def build_bm25_index_pyserini(self):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir, exist_ok=True)

        if self.corpus_path.endswith(".jsonl"):
            shutil.copyfile(self.corpus_path, temp_file_path)
            # check if the language is chinese
            with open(self.corpus_path, 'r', encoding='utf-8') as file:
                first_item = json.loads(file.readline())
                contents = first_item.get("contents", "")  # 获取 contents 字段
                zh_flag = judge_zh(contents)
        elif self.corpus_path.endswith(".parquet"):
            corpus = datasets.load_dataset('parquet', data_files=self.corpus_path, split="train")
            new_corpus = [{'id': idx, 'contents': text} for idx, text in enumerate(corpus['text'])]
            contents = new_corpus[0]['contents']
            zh_flag = judge_zh(contents)
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                for item in new_corpus:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        else:
            raise NotImplementedError

        print("Start building bm25 index...")
        pyserini_args = [
            "--collection",
            "JsonCollection",
            "--input",
            temp_dir,
            "--index",
            self.save_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
        ]

        if zh_flag:
            print("Use chinese bm25 mode")
            pyserini_args.append("--language")
            pyserini_args.append("zh")

        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)

        print("Finish!")

    def build_bm25_index_bm25s(self):
        """Building BM25 index based on bm25s library."""

        import bm25s
        import Stemmer

        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)

        corpus = load_corpus(self.corpus_path)
        # TODO: BM25s not support chinese well
        is_zh = judge_zh(corpus[0]['contents'])
        if is_zh:
            tokenizer = bm25s.tokenization.Tokenizer(stopwords='zh')
        else:
            stemmer = Stemmer.Stemmer("english")
            tokenizer = bm25s.tokenization.Tokenizer(stopwords='en', stemmer=stemmer)

        corpus_text = corpus["contents"]
        corpus_tokens = tokenizer.tokenize(corpus_text, return_as='tuple')
        retriever = bm25s.BM25(corpus=corpus, backend="numba")
        retriever.index(corpus_tokens)
        retriever.save(self.save_dir, corpus=None)
        tokenizer.save_vocab(self.save_dir)
        tokenizer.save_stopwords(self.save_dir)

        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        with open(embedding_path, 'rb') as f:
            reps, lookup = pickle.load(f)
            all_embeddings = np.array(reps).reshape(corpus_size, hidden_size)
        # all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(self.embedding_save_path, shape=all_embeddings.shape, mode="w+", dtype=all_embeddings.dtype)
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        encode_data = [item["contents"] for item in self.corpus]
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.batch_size = self.batch_size * self.gpu_num
            all_embeddings = self.encoder.multi_gpu_encode(encode_data, batch_size=self.batch_size, is_query=False)
        else:
            all_embeddings = self.encoder.encode(encode_data, batch_size=self.batch_size, is_query=False, dtype=self.torch_dtype)

        return all_embeddings

    def encode_all_clip(self):
        if self.index_modal == "all":
            modal_dict = {"text": None, "image": None}
        else:
            modal_dict = {self.index_modal: None}
        for modal, _ in modal_dict.items():
            encode_data = [item[modal] for item in self.corpus]
            if self.gpu_num > 1:
                print("Use multi gpu!")
                self.batch_size = self.batch_size * self.gpu_num
                all_embeddings = self.encoder.multi_gpu_encode(encode_data, batch_size=self.batch_size, modal=modal)
            else:
                all_embeddings = self.encoder.encode(encode_data, batch_size=self.batch_size, modal=modal)
            modal_dict[modal] = all_embeddings

        all_embeddings = np.concatenate(list(modal_dict.values()), axis=0)
        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and
        construct a faiss index.
        """

        if self.is_clip:
            from flashrag.retriever.encoder import ClipEncoder

            self.encoder = ClipEncoder(
                model_name=self.retrieval_method,
                model_path=self.model_path,
            )
            hidden_size = self.encoder.model.projection_dim

        elif self.use_sentence_transformer:
            from flashrag.retriever.encoder import STEncoder

            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.model_path,
                max_length=self.max_length,
                fp16=self.fp16,
                instruction=self.instruction,
            )
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
        else:
            # from flashrag.retriever.encoder import Encoder
            # self.encoder = Encoder(
            #     model_name=self.retrieval_method,
            #     model_path=self.model_path,
            #     pooling_method=self.pooling_method,
            #     max_length=self.max_length,
            #     use_fp16=self.fp16,
            #     instruction=self.instruction,
            # )
            from retriever import Encoder
            
            if self.training_args.bf16:
                self.torch_dtype = torch.bfloat16
            elif self.training_args.fp16:
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                (
                    self.model_args.tokenizer_name
                    if self.model_args.tokenizer_name
                    else self.model_args.model_name_or_path
                ),
                cache_dir=self.model_args.cache_dir,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if self.data_args.padding_side == "right":
                tokenizer.padding_side = "right"
            else:
                tokenizer.padding_side = "left"
            self.encoder = Encoder.load(
                self.model_args.model_name_or_path,
                pooling=self.model_args.pooling,
                normalize=self.model_args.normalize,
                lora_name_or_path=self.model_args.lora_name_or_path,
                cache_dir=self.model_args.cache_dir,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.model_args.attn_implementation,
                tokenizer=tokenizer,
                data_args=self.data_args,
                instruction=self.data_args.query_instruction,
            )
            self.encoder = self.encoder.to(self.training_args.device)
            self.encoder.eval()
            
            hidden_size = self.encoder.encoder.config.hidden_size

        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
            print(f"Loaded the provided embeddings from {self.embedding_path}")
        else:
            all_embeddings = self.encode_all_clip() if self.is_clip else self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        if self.is_clip:
            if self.index_modal == "all":
                assert all_embeddings.shape[0] % 2 == 0
                text_embedding = all_embeddings[: len(all_embeddings) // 2, :]
                image_embedding = all_embeddings[len(all_embeddings) // 2:, :]
                text_index_save_path = os.path.join(
                    self.save_dir, f"{self.retrieval_method}_{self.faiss_type}_text.index"
                )
                self.save_faiss_index(text_embedding, self.faiss_type, text_index_save_path)

                image_index_save_path = os.path.join(
                    self.save_dir, f"{self.retrieval_method}_{self.faiss_type}_image.index"
                )
                self.save_faiss_index(image_embedding, self.faiss_type, image_index_save_path)
            else:
                self.index_save_path = os.path.join(
                    self.save_dir, f"{self.retrieval_method}_{self.faiss_type}_{self.index_modal}.index"
                )
                self.save_faiss_index(all_embeddings, self.faiss_type, self.index_save_path)
        else:
            self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")
            if os.path.exists(self.index_save_path):
                print("The index file already exists and will be overwritten.")
            self.save_faiss_index(all_embeddings, self.faiss_type, self.index_save_path)
        print("Finish!")

    def save_faiss_index(
            self,
            all_embeddings,
            faiss_type,
            index_save_path,
    ):
        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, index_save_path)


def main():
    # parser = argparse.ArgumentParser(description="Creating index.")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # Basic parameters
    parser.add_argument("--retrieval_method", default="dense", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_embedded_path", type=str, default=None)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Parameters for building dense index
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)
    parser.add_argument("--bm25_backend", default="pyserini", choices=["bm25s", "pyserini"])

    # Parameters for build multi-modal retriever index
    parser.add_argument("--index_modal", type=str, default="all", choices=["text", "image", "all"])

    # New arguments for seismic index
    parser.add_argument("--n_postings", type=int, default=1000)
    parser.add_argument("--centroid_fraction", type=float, default=0.2)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    parser.add_argument("--summary_energy", type=float, default=0.4)
    parser.add_argument("--nknn", type=int, default=0)
    parser.add_argument("--batched_indexing", type=int, default=10000)

    # args = parser.parse_args()
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported.")
    
    args.model_path = model_args.model_name_or_path
    args.batch_size = training_args.per_device_eval_batch_size
    args.max_length = data_args.query_max_len
    args.pooling_method = model_args.pooling


    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=data_args.corpus_path,
        save_dir=args.save_dir,
        max_length=training_args.max_length,
        batch_size=args.batch_size,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        pooling_method=args.pooling_method,
        instruction=args.instruction,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        use_sentence_transformer=args.sentence_transformer,
        bm25_backend=args.bm25_backend,
        index_modal=args.index_modal,
        n_postings=args.n_postings,
        centroid_fraction=args.centroid_fraction,
        min_cluster_size=args.min_cluster_size,
        summary_energy=args.summary_energy,
        batched_indexing=args.batched_indexing,
        corpus_embedded_path=args.corpus_embedded_path,
        nknn=args.nknn,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
