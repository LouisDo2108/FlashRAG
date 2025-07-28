#!/bin/bash
#SBATCH --partition=fit
#SBATCH --account=ft49
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=fitq
#SBATCH --job-name=thuy0050
#SBATCH --output=/home/thuy0050/code/FlashRAG/logs/slurm-%x-%j.out
#SBATCH --error=/home/thuy0050/code/FlashRAG/logs/slurm-%x-%j.err
#SBATCH --time=1-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

#SBATCH --mail-user=tuan.huynh1@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ==== ENVIRONMENT SETUP ====
source ~/.bashrc
conda activate tevatron

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_USE_V1=0
# export TQDM_DISABLE=1 # Avoid logging tqdm progress bars
# export TORCH_USE_CUDA_DSA=0 # Set to 1 only if debugging
# export CUDA_LAUNCH_BLOCKING=0 # Set to 1 only if debugging

# Useful for pytorch debugging: torch.autograd.set_detect_anomaly(True)

cd /home/thuy0050/code/FlashRAG/scripts

DATA_ROOT_DIR=/home/thuy0050/mg61_scratch2/thuy0050/data/third_work
OUTPUT_DIR_ROOT=/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron

DATA_NAME=temporal_nobel_prize
MODEL_NAME=bge-base-en-v1.5
BACKBONE=bge-base-en-v1.5
EXP_NAME=bge-base-en-v1.5
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME

DATA_NAME=temporal_nobel_prize
MODEL_NAME=bge-base-en-v1.5 # ts-retriever
BACKBONE=bge-base-en-v1.5
EXP_NAME=bge-base-en-v1.5 # naive_temporal_v3_5epoch_temp0.05_lora_bf16
OUTPUT_DIR=$OUTPUT_DIR_ROOT/$DATA_NAME/$MODEL_NAME/$BACKBONE/$EXP_NAME

# CHECKPOINT_DIR=$OUTPUT_DIR 
CHECKPOINT_DIR=BAAI/bge-base-en-v1.5

mkdir -p $OUTPUT_DIR # Create folder if not exists

# This would simply load the corpus embedding from Tevatron inference, remember to provide embedding path!!!
python /home/thuy0050/code/FlashRAG/louis/index_builder.py \
  --model_path $CHECKPOINT_DIR \
  --corpus_path $DATA_ROOT_DIR/temporal/temporal_nobel_prize/test/corpus.jsonl \
  --save_dir $OUTPUT_DIR \
  --bf16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method cls \
  --faiss_type Flat \
  --embedding_path $OUTPUT_DIR/corpus_emb.pkl

python /home/thuy0050/code/FlashRAG/louis/simple_rag.py \
  --flashrag_config_yaml_path /home/thuy0050/code/FlashRAG/louis/simple_config.yaml \
  --per_device_eval_batch_size 512 \
  --query_max_len 512 \
  --pooling cls \
  --bf16 \
  --normalize \
  --attn_implementation sdpa \
  --encode_is_query \
  --query_instruction "Represent this sentence for searching relevant passages: " \
  --dataset_name LouisDo2108/temporal-nobel-prize \
  --dataset_config query \
  --model_name_or_path $CHECKPOINT_DIR \
  --encode_output_path $OUTPUT_DIR/queries_emb.pkl \
  --overwrite_output_dir
#   --lora_name_or_path $OUTPUT_DIR \