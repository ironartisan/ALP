import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,GPT2Tokenizer

from collections import defaultdict

from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity, find_layers
import sys
print('# of gpus: ', torch.cuda.device_count())


import json
import logging
import math

import random
from itertools import chain
from pathlib import Path

import datasets

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def main():


    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default="llm_weights/models--Enoch--llama-7b-hf/model", help='LLaMA model')
    parser.add_argument('--model', type=str, default="pruned/wanda/llama-7b-hf_sparsity_0.7", help='LLaMA model')
    parser.add_argument('--model_name', type=str, help='LLaMA model_name')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--grad_nsamples', type=int, default=1, help='grad_nsamples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="result", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha')
    parser.add_argument('--eval_zero_shot', action="store_true", help="whether to zero-shot eval")
    
    parser.add_argument(
    "--save_log", action="store_true", help="save log")
    
    
    args = parser.parse_args()


    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)



    model_name = args.model.split("/")[-1]
    args.model_name = model_name
    print(f"loading llm model {args.model}")
    
    


    
    accelerate=True
    task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "mnli", "openbookqa"]
    num_shot = 0
    
    results = eval_zero_shot(args.model, task_list, num_shot, accelerate)
    model_name = args.model.split("/")[-1]
    dirname = "eval_zero_shot"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('{}/results_zero_shot_{}.json'.format(dirname, model_name), 'a') as file:
        json.dump(results, file, indent=2)

if __name__ == '__main__':
    main()
