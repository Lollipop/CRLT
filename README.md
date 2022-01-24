
# CRLT: A Unified Contrastive Learning Toolkit for Unsupervised Text Representation Learning
This repository contains the code and relevant instructions of CRLT.

## Overview
The goal of CRLT is to provide an out-of-the-box toolkit for contrastive learning. Users only need to provide unlabeled data and edit a configuration file in the format of JSON, and then they can quickly train, use and evaluate representation learning models. CRLT consists of 6 critical modules, including data synthesis, negative sampling, representation encoders, learning paradigm, optimizing strategy and model evaluation. For each module, CRLT provides various popular implementations and therefore different kinds of CL architectures can be easily constructed using CRLT. 


## Installation

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```


Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```
### Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. See [our paper](https://arxiv.org/pdf/2104.08821.pdf) (Appendix B) for evaluation details.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Getting Started
