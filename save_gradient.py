import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,GPT2Tokenizer
from importlib.metadata import version
from transformers import AdamW
from lib.data import get_loaders
from lib.layer import Layer
import torch.nn as nn 
import argparse
import os


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids



def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        # cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    print("printing gpu allocation for all the layers")
    print(model.hf_device_map)
    model.seqlen = 2048
    return model

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngrad', type=int, default=10, help='no of samples used')
    parser.add_argument('--model', type=str,default="facebook/opt-6.7b", help='model to used') ## change meta-llama/Meta-Llama-3-8B
    args = parser.parse_args()
 
    
    model_args = args.model
    model_name = args.model.split("/")[-1]
    cache_dir_args = "llm_weights"
    args.model = cache_dir_args + "/models--" + args.model.replace("/", "--") + "/model"
    model = get_llm(args.model, cache_dir_args)
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)


    device=torch.device("cuda:0")
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    print("loading calibdation data")
    ngrad=args.ngrad
    seed=0
    dataloader, _ = get_loaders("c4",nsamples=ngrad,seed=seed,seqlen=64,tokenizer=tokenizer)
    print("dataset loading complete")
    optimizer = AdamW(model.parameters(), lr=0.01, eps=0.01)
    optimizer.zero_grad()

    grad_up = Layer(model, model_args)
    nsample = 0
    model.train()
    for input_ids, labels in dataloader:
        nsample+=1
        print("making gradient computation on sample: ", nsample)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels) 
        loss = outputs.loss
        print("Printing the loss:", loss)
        loss.backward()
        grad_up.update_layer(nsample)
        optimizer.zero_grad()
    print("Done")
    gradients_l2 = grad_up.total_conn_l2
    for name in gradients_l2:
        grad_sqrt = torch.sqrt(gradients_l2[name])
        gradients_l2[name] = grad_sqrt.to(dtype=torch.float16)
    
        
    
    if not os.path.exists(f'./gradients/{model_name}'):
        os.makedirs(f'./gradients/{model_name}')
    # with open(f'./gradients/{model_name}/gradientsnorm_l2_model_{model_name}_ngrad_{args.ngrad}.pth', 'wb') as f:
    #     torch.save(gradients_l2, f)
    # with open(f'./gradients/{model_name}/gradientsnorm_l1_model_{model_name}_ngrad_{args.ngrad}.pth', 'wb') as f:
    #     torch.save(grad_up.total_conn_l1, f)
    # with open(f'./gradients/{model_name}/gradientsnorm_l2_total_model_{model_name}_ngrad_{args.ngrad}.pth', 'wb') as f:
    #     torch.save(grad_up.total_conn_l2, f)
    # with open(f'./gradients/{model_name}/gradientsnorm_l2_ten_total_model_{model_name}_ngrad_{args.ngrad}.pth', 'wb') as f:
    #     torch.save(grad_up.total_conn_l2_ten, f)
    with open(f'./gradients/{model_name}/gradientsnorm_l2_hundred_total_model_{model_name}_ngrad_{args.ngrad}.pth', 'wb') as f:
        torch.save(grad_up.total_conn_l2_hundred, f)