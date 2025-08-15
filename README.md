#  ALP

Official PyTorch implementation of ALP: Adaptive Layerwise Pruning in Large Language Models
      
## Table of contents

* [Abstract](#abstract)
* [Results](#Results)
* [Installation](#installation)
* [Usage](#Usage)


## Abstract

Although large language models (LLMs) achieve strong performance on a wide range of downstream tasks, their massive parameter counts incur substantial computational and memory costs. One-shot unstructured pruning methods can remove a substantial proportion of redundant weights with minimal immediate retraining. However, these methods typically apply a uniform sparsity rate across all layers, ignoring inter-layer heterogeneity in importance and consequently suffering pronounced performance degradation at high sparsity levels. To overcome these limitations, we propose Adaptive Layerwise Pruning (ALP), an automatic method that allocates non-uniform per-layer sparsity by estimating the sensitivity of connections to the loss function using only ten calibration samples. ALP normalizes and aggregates per-connection sensitivities to derive a redundancy score for each layer, converts these scores into layer importance measures, and assigns sparsity in inverse proportion to importance. Extensive experiments show that ALP consistently outperforms both uniform and prior non-uniform baselines, particularly beyond 50% sparsity, and achieves up to a 3.2× CPU inference speedup at 80% sparsity while preserving model performance.


## Installation 
--- 
Installation instructions can be found in [INSTALL.md](INSTALL.md).



## Usage

### Calculating Connection Sensitivity

```
python save_gradient.py
```



### Script example of pruning llama-7b

```
    python   main.py \
    --model "Enoch/llama-7b-hf" \
    --grad_nsamples 10 \
    --alpha 0.15 \
    --prune_method "wanda_alp" \
    --sparsity_ratio 0.7 \
    --sparsity_type "unstructured" \
    --save_log
```

### Zero-shot Evaluation
```
    python   main.py \
    --model "Enoch/llama-7b-hf" \
    --grad_nsamples 10 \
    --alpha 0.15 \
    --prune_method "wanda_alp" \
    --sparsity_ratio 0.7 \
    --sparsity_type "unstructured" \
    --eval_zero_shot \
    --save_model "pruned/wanda_alp/llama-7b-hf_sparsity0.7" 
    --save_log
```

### Acknowledgement
The repository is build upon the [RIA](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning), [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repositories.




