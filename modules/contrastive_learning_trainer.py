from typing import Optional, List, Dict, Union, Callable, Tuple

import os
import torch
import senteval
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import datasets
import collections
from transformers.utils import logging
from torch.utils.data import Dataset, DataLoader
from transformers.trainer import TRAINING_ARGS_NAME
from transformers import Trainer, default_data_collator
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import is_datasets_available, WEIGHTS_NAME
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, denumpify_detensorize

from .loss.loss_utils import lalign, lunif

logger = logging.get_logger()


class ContrastiveLearningTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs
    ):
        super(ContrastiveLearningTrainer, self).__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers
        )

        self.senteval_data_path = kwargs.get(
            'senteval_data_path',
            "utils/SentEval/data"
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            raise ValueError("Term `loss` must be in the model output.")

        return (loss, outputs) if return_outputs else loss

    def evaluate_senteval(self, model, tasks: List[str], params: Dict) -> Dict:
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=64,
                truncation=True
            )
            for k in batch:
                batch[k] = self._prepare_inputs(batch[k])
            with torch.no_grad():
                outputs = model(
                    **batch,
                    return_sentence_embedding=True
                )
                sentence_embeddings = outputs.sentence_embeddings.data
            return sentence_embeddings.cpu()

        se = senteval.engine.SE(params, batcher, prepare)
        results = se.eval(tasks)
        return results

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        model.eval()
        self.callback_handler.eval_dataloader = dataloader

        if self.args.past_index >= 0:
            self._past = None

        # Set params for SentEval (fastmode)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        params = {
            'task_path': self.senteval_data_path,
            'usepytorch': True,
            'kfold': 5
        }
        params['classifier'] = {
            'nhid': 0,
            'optim': 'rmsprop',
            'batch_size': 128,
            'tenacity': 3,
            'epoch_size': 2
        }

        logger.info(f"***** Running SentEval {description} *****")
        logger.info(f"  Tasks = {', '.join(tasks)}.")
        logger.info(f"  Batch size = {params['classifier']['batch_size']}.")

        results = self.evaluate_senteval(model, tasks, params)
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2
        }

        logger.info(f"***** Running Align/ Uniform {description} *****")
        batch_size = dataloader.batch_size
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}.")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}.")

        embeddings_q, embeddings_k = None, None
        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            outputs = model(**inputs, return_sentence_embedding=True)

            embeddings_q = outputs.embeddings_q.data if embeddings_q is None else torch.cat(
                (embeddings_q, outputs.embeddings_q.data)
            )
            embeddings_k = outputs.embeddings_k.data if embeddings_k is None else torch.cat(
                (embeddings_k, outputs.embeddings_k.data)
            )

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        embeddings_q = F.normalize(embeddings_q.nan_to_num(), dim=1)
        embeddings_k = F.normalize(embeddings_k.nan_to_num(), dim=1)
        align = lalign(embeddings_q, embeddings_k).item()
        uniform = (lunif(embeddings_q).item() + lunif(embeddings_k).item()) / 2

        metrics.update({
            "align": align,
            "uniform": uniform,
            "avg_align_uniform": (align + self.args.lamda * uniform) / 2
        })

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=None)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        self._memory_tracker.start()
        dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(
            dataloader=dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation")

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=default_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model.encoder, PreTrainedModel):
            if isinstance(unwrap_model(self.model.encoder), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.encoder.state_dict()
                unwrap_model(self.model.encoder).save_pretrained(
                    output_dir, state_dict=state_dict)
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.encoder.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.encoder.load_state_dict(
            state_dict, strict=False)

        if len(load_result.missing_keys) != 0:
            if self.model.encoder._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model.encoder._keys_to_ignore_on_save
            ):
                self.model.encoder.tie_weights()
            else:
                logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")
