
from nltk.util import pr
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import Similarity, lalign, lunif


class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()
        self.similarity = Similarity(temperature)

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        if projected_q is not None and projected_k is not None:
            q, k = projected_q, projected_k

        batch_size = q.shape[0]
        similarity = self.similarity(
            q.unsqueeze(1), k.unsqueeze(0)
        )
        mask = torch.cat(
            [
                torch.ones_like(similarity, dtype=torch.bool),
                torch.ones_like(
                    similarity, dtype=torch.bool).fill_diagonal_(False)
            ], 1
        )
        similarity = torch.cat(
            [
                similarity,
                self.similarity(q.unsqueeze(1), q.unsqueeze(0))
            ], 1
        )
        similarity = similarity.masked_select(mask).view(batch_size, -1)
        if m is not None:
            similarity = torch.cat(
                [
                    similarity,
                    self.similarity(q.unsqueeze(1), m.unsqueeze(0))
                ], 1
            )
        labels = torch.arange(
            similarity.size(0),
            dtype=torch.long,
            device=similarity.device
        )
        return self.loss_fn(similarity, labels)


class Debiased(nn.Module):
    def __init__(self, temperature, tau_plus):
        super().__init__()
        self.tau_plus = tau_plus
        self.temperature = temperature
        self.similarity = Similarity(temperature)

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        if projected_q is not None and projected_k is not None:
            q, k = projected_q, projected_k

        batch_size = q.shape[0]

        z = torch.cat([q, k], 0)
        neg = torch.exp(
            self.similarity(z.unsqueeze(1), z.unsqueeze(0))
        )
        mask = torch.ones(size=(batch_size, batch_size),
                          dtype=torch.bool).fill_diagonal_(False)
        mask = mask.repeat(2, 2).cuda(neg.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        if m is not None:
            neg = torch.cat(
                [
                    neg,
                    self.similarity(z.unsqueeze(1), m.unsqueeze(0))
                ], 1
            )

        pos = torch.exp(self.similarity(q, k))
        pos = torch.cat([pos, pos], 0)

        N = 2 * (batch_size - 1)
        Ng = (-self.tau_plus * N * pos + neg.sum(dim=-1)) / (1 - self.tau_plus)
        Ng = torch.clamp(
            Ng,
            min=N * np.e ** (-1 / self.temperature)
        )
        return (-torch.log(pos / (pos + Ng))).mean()


class Hard(Debiased):
    def __init__(self, temperature, tau_plus, beta):
        super().__init__(temperature, tau_plus)
        self.beta = beta

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        if projected_q is not None and projected_k is not None:
            q, k = projected_q, projected_k

        batch_size = q.shape[0]

        z = torch.cat([q, k], 0)
        neg = torch.exp(
            self.similarity(z.unsqueeze(1), z.unsqueeze(0))
        )
        mask = torch.ones(size=(batch_size, batch_size),
                          dtype=torch.bool).fill_diagonal_(False)
        mask = mask.repeat(2, 2).cuda(neg.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        if m is not None:
            neg = torch.cat(
                [
                    neg,
                    self.similarity(z.unsqueeze(1), m.unsqueeze(0))
                ], 1
            )

        pos = torch.exp(self.similarity(q, k))
        pos = torch.cat([pos, pos], 0)

        N = 2 * (batch_size - 1)
        imp = (self.beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
        Ng = torch.clamp(
            Ng,
            min=N * np.e ** (-1 / self.temperature)
        )
        return (-torch.log(pos / (pos + Ng))).mean()


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity = nn.MSELoss()

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        # TODO: 这里应该是对称的函数,但是因为模型forward没有支持对称处理,所以只有单边.
        predicted_q = predicted_q.norm(p=2, dim=-1)
        projected_k = projected_k.detach().norm(p=2, dim=-1)

        return self.similarity(predicted_q, projected_k)


class AlignUniform(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        if projected_q is not None and projected_k is not None:
            q, k = projected_q, projected_k
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        align = lalign(q, k)
        if m is not None:
            m = F.normalize(m, dim=1)
            q = torch.vstack([q, m])
            k = torch.vstack([k, m])
        uniform = (lunif(q) + lunif(k)) / 2
        return align + self.lamda * uniform


class StopGradient(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        m=None
    ):
        return (self.D(predicted_q, projected_k) + self.D(predicted_k, projected_q)) / 2

    def D(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return - (p * z).sum(dim=1).mean()


class BarlowTwins(nn.Module):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def forward(
        self, q, k,
        projected_q=None,
        projected_k=None,
        predicted_q=None,
        predicted_k=None,
        n=None
    ):
        batch_size = predicted_q.shape[0]
        predicted_q_norm = (
            predicted_q - predicted_q.mean(0)) / predicted_q.std(0)
        predicted_k_norm = (
            predicted_k - predicted_k.mean(0)) / predicted_k.std(0)

        c = predicted_q_norm.T @ predicted_k_norm
        c.div_(batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        return on_diag + self.lamda * off_diag

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
