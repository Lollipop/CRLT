import os
import sys
import time
import random
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass, field

import torch
import datasets
from datasets.utils import DownloadConfig
from datasets import load_dataset, load_metric
import transformers
from transformers import (
    set_seed,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5EncoderModel,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    DataCollatorWithPadding,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint

from modules.generator.generator import Generator
from modules.contrastive_learning_trainer import ContrastiveLearningTrainer
from modules.contrastive_learning_for_representation_learning import ContrastiveLearningForRepresentationLearning

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class UtilArguments:
    proxies: Optional[int] = field(default=7890)

    def __post_init__(self):
        if self.proxies is not None:
            self.proxies = {
                "http": "127.0.0.1:%s" % self.proxies,
                "https": "127.0.0.1:%s" % self.proxies
            }


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """

    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv file containing the training data."
        }
    )
    validation_file: Optional[str] = field(
        default="data/STSB/stsb_above_4.csv",
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv file)."},
    )
    senteval_data_path: Optional[str] = field(
        default="utils/SentEval/data",
        metadata={
            "help": "File path of SentEval library data."
        }
    )
    max_seq_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default="cache/datasets",
        metadata={
            "help": "Where do you want to store the processed data."
        }
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        }
    )
    # TODO: the setting must be true.
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # TODO: may need a test.
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    generator_params: Optional[Dict] = field(
        default=None,
        metadata={
            "help": "Parameters used to control how to generate query/key sample."
        }
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension == "csv", "`train_file` should be a csv file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "csv", "`validation_file` should be a csv file."

        dataset_name = self.train_file.split("/")[-1].split(".")[0]
        self.dataset_cache_dir = os.path.join(
            self.dataset_cache_dir,
            dataset_name
        )

        if self.generator_params is None:
            self.generator_params = {
                "query": None,
                "key": None
            }
        else:
            assert "query" in self.generator_params, "Term `query` cannot be found in the `generator_params`."
            # If `key` is not setted, it will use the same setting as query.
            if "key" not in self.generator_params:
                self.generator_params["key"] = self.generator_params["query"]


@ dataclass
class ModelTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default="/share/model/transformers/bert/uncased_L-12_H-768_A-12",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    model_cache_dir: Optional[str] = field(
        default="cache/models",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    hidden_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        }
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )

    extractor_params: Optional[Dict] = field(
        default=None,
        metadata={
            "help": "Parameters used to control how the representation is obtained from the model output"
        }
    )

    def __post_init__(self):
        model_name = self.model_name_or_path.split("/")[-1]
        self.model_cache_dir = os.path.join(
            self.model_cache_dir,
            model_name
        )
        VALID_EXTRACTOR_PARAMS = ["sentence_representation", ]
        extractor = list(self.extractor_params.keys())[0]
        assert extractor in VALID_EXTRACTOR_PARAMS, \
            "Not supported extractor setting."


@dataclass
class ContrastiveLearningTrainingArguments(TrainingArguments):
    use_negative: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If hard negtive is used, a `negative` column must in the `train_file`."
        }
    )
    use_siamese_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use siamese model as encoder for `key` samples."
        }
    )
    use_momentum: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use momentum update for siamese network."
        }
    )
    gamma: Optional[float] = field(
        default=0.999,
        metadata={
            "help": "Momentum update coefficient."
        }
    )
    memory_bank_size: int = field(
        default=0,
        metadata={
            "help": "If `memory_bank_size` is not 0, use memory bank for more key samples like MoCo."
        }
    )
    loss_fn: Optional[str] = field(
        default="infonce",
        metadata={
            "help": "TODO"
        }
    )
    temperature: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Hyperparameter in the loss function."
        }
    )
    tau_plus: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Hyperparameter in the loss function."
        }
    )
    beta: Optional[float] = field(
        default=1,
        metadata={
            "help": "Hyperparameter in the loss function."
        }
    )
    lamda: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Hyperparameter in the `align_uniform` loss function."
        }
    )
    early_stopping_patience: Optional[int] = field(
        default=15,
        metadata={
            "help": "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls."
        }
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelTrainingArguments, DataTrainingArguments, ContrastiveLearningTrainingArguments, UtilArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, util_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, util_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can provide your own CSV training and evaluation files (see below)
    # If the CSVs contain only one non-label column, the script does contrastive learning on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Loading a dataset from your local files. CSV training (and evaluation files) is needed.
    data_files = {"train": data_args.train_file}
    # Get the eval dataset: you can provide your own CSV test file (see below)
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    download_config = DownloadConfig(
        cache_dir=data_args.dataset_cache_dir,
        proxies=util_args.proxies if util_args.proxies is not None else None
    )
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_train_dataset = load_dataset(
            "csv",
            data_files=data_files["train"],
            cache_dir=data_args.dataset_cache_dir,
            download_config=download_config
        )
        if data_args.validation_file is not None:
            raw_validation_dataset = load_dataset(
                "csv",
                data_files=data_files["validation"],
                cache_dir=data_args.dataset_cache_dir,
                download_config=download_config
            )
    else:
        raise NotImplementedError("Other file types are not yet supported.")

    raw_train_dataset, raw_validation_dataset = \
        raw_train_dataset["train"], raw_validation_dataset["train"]

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config, unused_kwargs = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        proxies=util_args.proxies if util_args.proxies is not None else None,
        hidden_dropout_prob=model_args.hidden_dropout_prob,
        attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
        return_unused_kwargs=True
    )
    config.update(unused_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        proxies=util_args.proxies if util_args.proxies is not None else None
    )

    # TODO: check if use the correct column.
    non_label_column_names = [
        name for name in raw_train_dataset.column_names if name != "label"]
    if training_args.use_negative:
        assert "negative" in non_label_column_names, "`train_file` should contain a `negative` column"
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        raise NotImplementedError("Padding strategy has not been implemented.")

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    query_generator = Generator(data_args.generator_params["query"])
    key_generator = Generator(data_args.generator_params["key"])

    def preprocess_function(examples):
        batch_size = len(examples[sentence1_key])

        if sentence2_key is None:
            query_sentences = query_generator.preprocess(
                examples[sentence1_key]
            )

            query_features = tokenizer(
                query_sentences,
                padding=padding,
                max_length=max_seq_length,
                truncation=True
            )
            query_features = query_generator.postprocess(query_features)

            key_sentences = key_generator.preprocess(
                examples[sentence1_key])
            key_features = tokenizer(
                key_sentences,
                padding=padding,
                max_length=max_seq_length,
                truncation=True
            )
            key_features = key_generator.postprocess(key_features)
            features = {}
            for key in query_features:
                features[key] = [
                    [
                        query_features[key][i],
                        key_features[key][i]
                    ] for i in range(batch_size)
                ]
        else:
            sentences = examples[sentence1_key] + examples[sentence2_key]
            features = tokenizer(
                sentences,
                padding=padding,
                max_length=max_seq_length,
                truncation=True
            )
            for key in features:
                features[key] = [
                    [
                        features[key][i],
                        features[key][i+batch_size]
                    ]
                    for i in range(batch_size)
                ]

        if training_args.use_negative:
            neg_features = tokenizer(
                examples["negative"],
                padding=padding,
                max_length=max_seq_length,
                truncation=True
            )
            for key in features:
                for i in range(batch_size):
                    features[key][i].append(neg_features[key][i])

        return features

    def process_function_eval(examples):
        batch_size = len(examples["sentence1"])
        sentences = examples["sentence1"] + examples["sentence2"]
        features = tokenizer(
            sentences,
            padding=padding,
            max_length=max_seq_length,
            truncation=True
        )
        for key in features:
            features[key] = [
                [
                    features[key][i],
                    features[key][i+batch_size]
                ]
                for i in range(batch_size)
            ]
        return features

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_train_dataset = raw_train_dataset.select(
                range(data_args.max_train_samples))
        with training_args.main_process_first(desc="dataset map pre-processing"):
            train_dataset = raw_train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_train_dataset.column_names,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_validation_dataset = raw_validation_dataset.select(
                range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = raw_validation_dataset.map(
                process_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_validation_dataset.column_names,
                desc="Running tokenizer on validation dataset",
            )

    if sentence2_key is None:
        query_generator_methods = ",".join(
            data_args.generator_params["query"].keys()) if data_args.generator_params["query"] is not None else "dropout"
        key_generator_methods = ",".join(
            data_args.generator_params["key"].keys()) if data_args.generator_params["key"] is not None else "dropout"
        logger.info(
            f"Using `{query_generator_methods}` to create query examples of column `{sentence1_key}`."
        )
        logger.info(
            f"Using `{key_generator_methods}` to create key examples of column `{sentence1_key}`."
        )
    else:
        logger.info(
            f"Using column {sentence1_key} and {sentence2_key} as pair data to build different view examples."
        )
    if training_args.use_negative:
        logger.info(
            f"Using column `negative` as hard negative sample."
        )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: ")
            for k, v in train_dataset[index].items():
                logger.info(
                    f"{k} of the training set: ")
                logger.info(f"{v[0]}")
                logger.info(f"{v[1]}")
                if training_args.use_negative:
                    logger.info(f"{v[2]}")

    # Data collator will default to default_data_collator, so we change it if we didn't do the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    # TODO: this may accelerate the training progress!!
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
        raise NotImplementedError("Do not support fp16 training.")
    else:
        raise NotImplementedError

    model = ContrastiveLearningForRepresentationLearning(
        generator=(query_generator, key_generator),
        config=config,
        pretrained_model_name_or_path=model_args.model_name_or_path,
        extractor_params=model_args.extractor_params,
        use_siamese_encoder=training_args.use_siamese_encoder,
        use_momentum=training_args.use_momentum,
        gamma=training_args.gamma,
        memory_bank_size=training_args.memory_bank_size,
        loss_fn=training_args.loss_fn,
        temperature=training_args.temperature,
        tau_plus=training_args.tau_plus,
        beta=training_args.beta,
        lamda=training_args.lamda,
        cache_dir=model_args.model_cache_dir,
        proxies=util_args.proxies,
    )

    # Initialize our Trainer
    trainer = ContrastiveLearningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(training_args.early_stopping_patience)
        ],
        senteval_data_path=data_args.senteval_data_path
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # TODO: not support resume from chechpoint
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        import senteval
        from prettytable import PrettyTable

        logger.info("*** Predict ***")
        start_time = time.time()
        tasks = [
            'STS12', 'STS13', 'STS14', 'STS15',
            'STS16', 'STSBenchmark', 'SICKRelatedness'
        ]
        if config.model_type == "t5":
            predict_model = T5EncoderModel.from_pretrained(
                training_args.output_dir).cuda().eval()
        else:
            predict_model = AutoModel.from_pretrained(
                training_args.output_dir).cuda().eval()

        tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)
        params = {
            'task_path': data_args.senteval_data_path,
            'usepytorch': True,
            'kfold': 10
        }
        params['classifier'] = {
            'nhid': 0,
            'optim': 'adam',
            'batch_size': 64,
            'tenacity': 5,
            'epoch_size': 4
        }

        def prepare(params, samples):
            return

        def batcher(params, batch):
            # Handle rare token encoding issues in the dataset
            if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
                batch = [[word.decode('utf-8') for word in s] for s in batch]

            sentences = [' '.join(s) for s in batch]

            # Tokenization
            inputs = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

            for k, v in inputs.items():
                inputs[k] = v.cuda()

            with torch.no_grad():
                outputs = predict_model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            representation = model.extractor(
                inputs["attention_mask"],
                outputs,
                return_sentence_embedding=True
            )
            return representation.data.cpu()

        results = {}
        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append(
                        "%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append(
                        "%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score)
                      for score in scores]) / len(scores)))
        tabel = PrettyTable()
        tabel.field_names = task_names
        tabel.add_row(scores)
        logger.info(f"\n{tabel}\n")

        output_predict_file = os.path.join(
            training_args.output_dir, f"predict_results.csv")

        metrics = {k: v for k, v in zip(task_names, scores)}
        metrics = pd.DataFrame(metrics, index=[0])
        metrics.to_csv(output_predict_file, index=False)

        secs = time.time() - start_time
        msec = int(abs(secs - int(secs)) * 100)
        logger.info("***** predict metrics ***** ")
        logger.info(
            f"prediction run_time: {datetime.timedelta(seconds=int(secs))}.{msec:02d}"
        )


if __name__ == "__main__":
    main()
