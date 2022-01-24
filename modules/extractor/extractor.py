from typing import Dict
import torch.nn as nn


class SentenceRepresentationExtractor(nn.Module):
    def __init__(self, config, params):
        super().__init__()
        self.pooler = params["pooler"]

        projector_params = params.get("projector", None)
        predictor_params = params.get("predictor", None)

        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size,  projector_params["hidden_size"]),
            nn.Tanh()
        ) if projector_params is not None else None

        if projector_params is None:
            assert predictor_params is None, "If projector is not used, predict also shouldn't be used."

        self.predictor = nn.Sequential(
            nn.Linear(
                projector_params["hidden_size"],
                predictor_params["hidden_size"],
                bias=False
            ),
            nn.BatchNorm1d(predictor_params["hidden_size"]),
            nn.ReLU(),
            nn.Linear(
                predictor_params["hidden_size"],
                config.hidden_size
            )
        ) if predictor_params is not None else None

    def forward(self, attention_mask, features, return_sentence_embedding=False):
        last_hidden_state = features.last_hidden_state * \
            attention_mask.unsqueeze(-1)

        if self.pooler == "cls":
            embeddings = last_hidden_state[:, 0]
        elif self.pooler == "avg":
            embeddings = last_hidden_state.sum(
                1) / (attention_mask.sum(-1)).unsqueeze(-1)
        elif self.pooler == "avg_first_last":
            hidden_states = features.hidden_states
            first_hidden_state = hidden_states[0] * \
                attention_mask.unsqueeze(-1)
            embeddings = (first_hidden_state + last_hidden_state).sum(1) / \
                (attention_mask.sum(-1)).unsqueeze(-1) / 2

        if return_sentence_embedding:
            return embeddings

        projected_embeddings = self.projector(
            embeddings) if self.projector is not None else None
        predicted_embeddings = self.predictor(
            projected_embeddings) if self.predictor is not None else None

        return embeddings, projected_embeddings, predicted_embeddings
