#!/bin/sh


export HF_ENDPOINT=https://hf-mirror.com

# model="meta-llama/Llama-2-7b-hf"
# model="Enoch/llama-7b-hf"
model="../pruned/llama-7b-hf_sparsegpt_csl_0.7"
model_name=$(echo "$model" | awk -F'/' '{print $2}')

lora_ft_model_path="${model_name}_lora_sparsegpt_csl_0.7"

echo "lora output dir:$lora_ft_model_path"

mkdir -p ${lora_ft_model_path}


export CUDA_VISIBLE_DEVICES=0

python finetune_lm.py \
    --model_name_or_path "${model}" \
    --config_name "${model}" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 240000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir "${lora_ft_model_path}" > ft_${model_name}_sparsegpt_csl_0.7.log
