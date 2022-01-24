
# CRLT: A Unified Contrastive Learning Toolkit for Unsupervised Text Representation Learning
This repository contains the code and relevant instructions of CRLT.

## Overview
The goal of CRLT is to provide an out-of-the-box toolkit for contrastive learning. Users only need to provide unlabeled data and edit a configuration file in the format of JSON, and then they can quickly train, use and evaluate representation learning models. CRLT consists of 6 critical modules, including data synthesis, negative sampling, representation encoders, learning paradigm, optimizing strategy and model evaluation. For each module, CRLT provides various popular implementations and therefore different kinds of CL architectures can be easily constructed using CRLT. 

![framework](./materials/flow.svg)


## Installation

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). Please use the correct `1.7.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. 

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

The evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation. See [SimCSE](https://arxiv.org/pdf/2104.08821.pdf) for more details.

Before training, please download the relevent datasets by running:
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Getting Started

### Data

For unsupervised SimCSE, we sample 1 million sentences from English Wikipedia; for supervised SimCSE, we use the SNLI and MNLI datasets. You can run data/download_wiki.sh and data/download_nli.sh to download the two datasets.

### Training

#### GUI
We provide example training scripts for SimCSE (the unsupervised version) by running:
```bash
conda activate crlt
python app.py
```
After editing the training parmameters, users click the `RUN` botton and will get the evaluation result in the same page.

#### Terminal
Rather than trainging with the web GUI, users can alse training by running:
```bash
python main.py examples/simcse.json
```

