from typing import Union, Dict

import torch
import torch.nn as nn
from .model_utils import (
    get_encoder,
    get_loss_fn,
    get_extractor,
    ContrastiveLearningOutput
)


class ContrastiveLearningForRepresentationLearning(nn.Module):
    def __init__(
        self,
        generator,
        config,
        pretrained_model_name_or_path: str,
        extractor_params: Dict,
        use_siamese_encoder: bool = False,
        use_momentum: bool = False,
        gamma: float = 0.999,
        memory_bank_size: int = 0,
        loss_fn: Union[nn.Module, str] = "infonce",
        temperature: float = 0.05,
        tau_plus: float = 1.0,
        beta: float = 1.0,
        lamda: float = 0.5,
        ** kwargs
    ):
        super().__init__()
        self.config = config

        encoder = get_encoder(config)
        self.encoder = encoder.from_pretrained(
            config=config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            generator=generator[0],
            add_pooling_layer=False,
            cache_dir=kwargs.get("cache_dir", None),
            proxies=kwargs.get("proxies", None),
        )
        self.extractor, projector_hidden_size = get_extractor(
            config, extractor_params)

        self.siamese_encoder = self.encoder
        self.siamese_extractor = self.extractor

        if use_siamese_encoder:
            self.siamese_encoder = encoder.from_pretrained(
                config=config,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                generator=generator[1],
                add_pooling_layer=False,
                cache_dir=kwargs.get("cache_dir", None),
                proxies=kwargs.get("proxies", None),
            )
            self.siamese_extractor, _ = get_extractor(config, extractor_params)

        self.gamma = gamma
        self.use_momentum = use_momentum

        if self.use_momentum and self.siamese_encoder == self.encoder:
            raise ValueError(
                "When use momentum update, the `use_siamese_encoder` must be TRUE."
            )

        if memory_bank_size != 0:
            self.memory_bank_size = memory_bank_size

            if projector_hidden_size is not None:
                memory_bank_dim = projector_hidden_size
            else:
                memory_bank_dim = config.hidden_size

            self.register_buffer(
                "memory_bank",
                torch.randn(memory_bank_size, memory_bank_dim)
            )
            self.register_buffer(
                "memory_bank_ptr",
                torch.zeros(1, dtype=torch.long)
            )
        else:
            self.memory_bank = None

        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_fn(
                loss_fn=loss_fn,
                temperature=temperature,
                tau_plus=tau_plus,
                beta=beta,
                lamda=lamda
            )
        else:
            self.loss_fn = loss_fn

        self._init_siamese_encoder(use_siamese_encoder)

    def _init_siamese_encoder(self, use_siamese_encoder=False):
        if use_siamese_encoder:
            for param_a, param_b in zip(self.encoder.parameters(), self.siamese_encoder.parameters()):
                param_b.data.copy_(param_a.data)
                if self.use_momentum:
                    param_b.requires_grad = False

            for param_a, param_b in zip(self.extractor.parameters(), self.siamese_extractor.parameters()):
                param_b.data.copy_(param_a.data)
                if self.use_momentum:
                    param_b.requires_grad = False

    @torch.no_grad()
    def _momentum_update_siamese_encoder(self):
        """
        Momentum update of the siamese encoder.
        """
        if self.use_momentum:
            for param_a, param_b in zip(self.encoder.parameters(), self.siamese_encoder.parameters()):
                param_b.data = \
                    param_b.data * self.gamma + \
                    param_a.data * (1. - self.gamma)
            for param_a, param_b in zip(self.extractor.parameters(), self.siamese_extractor.parameters()):
                param_b.data = \
                    param_b.data * self.gamma + \
                    param_a.data * (1. - self.gamma)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.memory_bank_ptr)
        assert self.memory_bank_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.memory_bank[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.memory_bank_size  # move pointer

        self.memory_bank_ptr[0] = ptr

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        return_sentence_embedding=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sentence_num = input_ids.size(1) if len(input_ids.shape) == 3 else 1

        if not return_sentence_embedding:
            input_ids_q, input_ids_k = \
                input_ids[:, 0, :], input_ids[:, 1, :]
            attention_mask_q, attention_mask_k = \
                attention_mask[:, 0, :], attention_mask[:, 1, :]
            if token_type_ids is not None:
                token_type_ids_q, token_type_ids_k = \
                    token_type_ids[:, 0, :], token_type_ids[:, 1, :]
            else:
                token_type_ids_q, token_type_ids_k = None, None

            if sentence_num > 2:
                input_ids_n = input_ids[:, 2, :]
                attention_mask_n = attention_mask[:, 2, :]
                if token_type_ids is not None:
                    token_type_ids_n = token_type_ids[:, 2, :]
                else:
                    token_type_ids_n = None, None
        else:
            if sentence_num > 1:
                input_ids = input_ids.view(
                    (-1, input_ids.size(-1)))  # (bs * num_sent, len)
                attention_mask = attention_mask.view(
                    (-1, attention_mask.size(-1)))  # (bs * num_sent len)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.view(
                        (-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        if not return_sentence_embedding:
            # If use momentum update, then update the siamese encoder.
            self._momentum_update_siamese_encoder()

            outputs = self.encoder(
                input_ids_q,
                attention_mask=attention_mask_q,
                token_type_ids=token_type_ids_q
            )
            embeddings_q, projected_embeddings_q, predicted_embeddings_q = self.extractor(
                attention_mask_q, outputs)

            if self.use_momentum:
                torch.set_grad_enabled(False)

            outputs = self.siamese_encoder(
                input_ids_k,
                attention_mask_k,
                token_type_ids_k
            )
            embeddings_k, projected_embeddings_k, predicted_embeddings_k = self.siamese_extractor(
                attention_mask_k, outputs)

            if sentence_num > 2:
                outputs = self.siamese_encoder(
                    input_ids_n,
                    attention_mask_n,
                    token_type_ids_n
                )
                embeddings_n, projected_embeddings_n, _ = self.siamese_extractor(
                    attention_mask_n, outputs
                )
            else:
                embeddings_n, projected_embeddings_n = None, None

            if self.use_momentum:
                torch.set_grad_enabled(True)
        else:
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_sentence_embedding=return_sentence_embedding
            )
            embeddings = self.extractor(
                attention_mask, outputs, return_sentence_embedding
            )

        if not return_sentence_embedding:
            if self.memory_bank is not None:
                if projected_embeddings_n is not None:
                    embeddings_m = torch.vstack(
                        [
                            projected_embeddings_n,
                            self.memory_bank.clone().detach()
                        ]
                    )
                elif embeddings_n is not None:
                    embeddings_m = torch.vstack(
                        [
                            embeddings_n,
                            self.memory_bank.clone().detach()
                        ]
                    )
                else:
                    embeddings_m = self.memory_bank.clone().detach()
            else:
                embeddings_m = projected_embeddings_n if projected_embeddings_n is not None else embeddings_n

            loss = self.loss_fn(
                embeddings_q,
                embeddings_k,
                projected_embeddings_q,
                projected_embeddings_k,
                predicted_embeddings_q,
                predicted_embeddings_k,
                embeddings_m
            )
            if self.memory_bank is not None:
                if projected_embeddings_k is not None:
                    self._dequeue_and_enqueue(projected_embeddings_k)
                elif embeddings_k is not None:
                    self._dequeue_and_enqueue(embeddings_k)

            return ContrastiveLearningOutput(
                loss=loss,
                embeddings_q=embeddings_q,
                embeddings_k=embeddings_k,
                embeddings_m=embeddings_m
            )
        else:
            if sentence_num > 1:
                embeddings = embeddings.view(
                    -1, sentence_num, embeddings.size(-1)
                )
                return ContrastiveLearningOutput(
                    embeddings_q=embeddings[:, 0],
                    embeddings_k=embeddings[:, 1]
                )
            return ContrastiveLearningOutput(
                sentence_embeddings=embeddings
            )
