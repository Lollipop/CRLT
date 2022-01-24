import torch
import torch.nn as nn


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, t):
        super().__init__()
        self.t = t
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.t


def lalign(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def lunif(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
