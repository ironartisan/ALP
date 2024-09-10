import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer,GPT2Tokenizer
from lib.prune_all import  prune_wanda,prune_magnitude,prune_sparsegpt,prune_wanda_alp,prune_sparsegpt_alp,prune_mag_alp,get_per_layer_ratio,prune_ria,prune_ria_alp
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity
import sys
print('# of gpus: ', torch.cuda.device_count())
from pdb import set_trace as st


import json
from accelerate.logging import get_logger


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        # cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def main():


    ########################## pruning ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_name', type=str, help='LLaMA model_name')
    parser.add_argument('--grad_type', type=str, default='l2_hundred_total',help='Type of  gradient') # l1,l2,l2_total,l2_ten_total,l2_hundred_total
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--grad_nsamples', type=int, default=10, help='grad_nsamples')
    parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured")
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="result", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--alpha', type=float, default=0.15, help='alpha')
    parser.add_argument('--eval_zero_shot', action="store_true", help="whether to zero-shot eval")
    parser.add_argument('--dataset', type=str,default="wikitext2",help="The name of the dataset to use.")
    parser.add_argument("--save_log", action="store_true", help="save log")
    ########################## gptq ################################
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument('--wbits', type=int, default=16,help='Whether to quantize as well.')
    parser.add_argument('--sym', action='store_true',help='Whether to perform symmetric quantization.')
    parser.add_argument('--percdamp', type=float, default=.01,help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--groupsize', type=int, default=-1,help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--act-order', action='store_true',help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--static-groups', action='store_true',help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.')
    
    ########################## gptq ################################
    
    ########################## RIA ################################
    parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")
    ########################## RIA ################################
    
    
    args = parser.parse_args()

    
    print("args.alpha",args.alpha)
    print ("args.nsamples",args.nsamples)
    print ("args.dataset",args.dataset)
    print ("args.prune_method",args.prune_method)
    
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print("args.sparsity_type",args.sparsity_type)
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))



    model_name = args.model.split("/")[-1]
    args.model_name = model_name

    print("args.alpha",args.grad_type)
    args.grad_path = f'./gradients/{model_name}/gradientsnorm_{args.grad_type}_model_{model_name}_ngrad_{args.grad_nsamples}.pth'

    if "alp" in args.prune_method:
        ratios = get_per_layer_ratio(args)
    print(f"loading llm model {args.model}")
    
    # Offline load moodel
    args.model = args.cache_dir + "/models--" + args.model.replace("/", "--") + "/model"

    model = get_llm(args.model, args.cache_dir)
    
    print ("model is =================================================================================")
    print (model.__class__.__name__)
    print (model)
    
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    print ("target sparsity", args.sparsity_ratio)   
       
    model.eval()

    print("pruning starts")


    ############################ baseline   ############################
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m) 
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "ria":
        prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ############################ alp   ############################
    elif args.prune_method == "wanda_alp":
        prune_wanda_alp(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt_alp":
        prune_sparsegpt_alp(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude_alp":
        prune_mag_alp(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "ria_alp":
        prune_ria_alp(args, model, tokenizer, ratios, device,  prune_n=prune_n, prune_m=prune_m)
    ############################ alp   ############################

    elif args.prune_method == "dense":
        pass


    print(f" prune method is {args.prune_method}")  
    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(model, tokenizer, device, dataset=args.dataset)
    print(f"ppl on {args.dataset} {ppl_test}")

    sys.stdout.flush()

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")

    if args.save_log:
        dirname = "results/{}".format(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        filename = f"log_{args.prune_method}_.txt"
        save_filepath = os.path.join(dirname, filename)
        with open(save_filepath, "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\talpha\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.alpha}\t{ppl_test:.4f}", file=f, flush=True)
                
    if args.eval_zero_shot:
        accelerate=True
        task_list = ["boolq", "rte", "hellaswag", "arc_challenge", "mnli",  "openbookqa"]
        num_shot = 0
        
        
        if args.save_model:
            eval_model = args.save_model
        else:
            eval_model = args.model
        results = eval_zero_shot(eval_model, task_list, num_shot, accelerate)
        model_name = eval_model.split("/")[-1]
        dirname = "eval_zero_shot"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open('{}/results_zero_shot_{}.json'.format(dirname, model_name), 'a') as file:
            json.dump(results, file, indent=2)
    
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
