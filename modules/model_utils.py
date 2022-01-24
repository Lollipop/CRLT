from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from .loss.losses import (
    InfoNCE,
    Debiased,
    Hard,
    MSE,
    AlignUniform,
    StopGradient,
    BarlowTwins
)
from .models.bert_model_for_contrastive_learning import BertModelForContrastiveLearning
from .models.roberta_model_for_contrastive_learning import RobertaModelForContrastiveLearning
from .models.albert_model_for_contrastive_learning import AlbertModelForContrastiveLearning
from .models.bigbird_model_for_contrastive_learning import BigBirdModelForContrastiveLearning
from .models.xlnet_model_for_contrastive_learning import XLNetModelForContrastiveLearning
from .models.electra_model_for_contrastive_learning import ElectraModelForContrastiveLearning
from .models.t5_encoder_model_for_contrastive_learning import T5EncoderModelForContrastiveLearning
from .models.bart_model_for_contrastive_learning import BartModelForContrastiveLearning
from .extractor.extractor import SentenceRepresentationExtractor


@dataclass
class ContrastiveLearningOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    embeddings_q: Optional[Tuple[torch.FloatTensor]] = None
    embeddings_k: Optional[Tuple[torch.FloatTensor]] = None
    embeddings_m: Optional[Tuple[torch.FloatTensor]] = None
    sentence_embeddings: Optional[Tuple[torch.FloatTensor]] = None


def get_extractor(config, params):
    extractor = list(params.keys())[0]

    if "projector" in params[extractor] and params[extractor]["projector"] is not None:
        projector_hidden_size = params[extractor]["projector"].get(
            "hidden_size",
            None
        )
    else:
        projector_hidden_size = None

    if extractor == "sentence_representation":
        return SentenceRepresentationExtractor(config, params[extractor]), projector_hidden_size


def get_loss_fn(loss_fn, temperature, tau_plus, beta, lamda):
    if loss_fn == "infonce":
        return InfoNCE(temperature=temperature)
    elif loss_fn == "debiased":
        return Debiased(temperature=temperature, tau_plus=tau_plus)
    elif loss_fn == "hard":
        return Hard(temperature=temperature, tau_plus=tau_plus, beta=beta)
    elif loss_fn == "mse":
        return MSE()
    elif loss_fn == "align_uniform":
        return AlignUniform(lamda=lamda)
    elif loss_fn == "stop_gradient":
        return StopGradient()
    elif loss_fn == "barlow_twins":
        return BarlowTwins(lamda=lamda)


def get_encoder(config):
    if config.model_type == "bert":
        return BertModelForContrastiveLearning
    elif config.model_type == "roberta":
        return RobertaModelForContrastiveLearning
    elif config.model_type == "albert":
        return AlbertModelForContrastiveLearning
    elif config.model_type == "big_bird":
        return BigBirdModelForContrastiveLearning
    elif config.model_type == "xlnet":
        return XLNetModelForContrastiveLearning
    elif "electra" in config.architectures[0].lower():
        return ElectraModelForContrastiveLearning
    elif config.model_type == "t5":
        return T5EncoderModelForContrastiveLearning
    elif config.model_type == "bart":
        return BartModelForContrastiveLearning
    else:
        raise NotImplementedError("Not support model type.")
