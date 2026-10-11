"""Ranker and retriever architectures.

All rankers consume `x: int64[B, F]` holding per-field local indices; the
field offsets into the shared embedding table live inside the module so the
exported ONNX graph takes the same tensor the data pipeline produces.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FieldEmbedding(nn.Module):
    def __init__(self, field_dims, dim: int):
        super().__init__()
        self.emb = nn.Embedding(int(sum(field_dims)), dim)
        offsets = np.concatenate([[0], np.cumsum(field_dims)[:-1]]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        nn.init.normal_(self.emb.weight, std=1e-2 if dim > 1 else 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x + self.offsets)


def mlp(d_in: int, hidden, dropout: float, out: int | None = 1, bn: bool = True) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h)]
        if bn:
            layers += [nn.BatchNorm1d(h)]
        layers += [nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    if out is not None:
        layers += [nn.Linear(d_in, out)]
    return nn.Sequential(*layers)


class CTRModel(nn.Module):
    """Linear term + optional FM second-order term + optional deep tower.

    lr:      fm=False, dnn=False
    fm:      fm=True,  dnn=False
    dnn:     fm=False, dnn=True   (wide & deep style)
    deepfm:  fm=True,  dnn=True   (Guo et al., 2017 - FM and DNN share embeddings)
    """

    def __init__(self, field_dims, embed_dim=16, hidden=(400, 400, 400), dropout=0.2,
                 fm=True, dnn=True, output_prob=False):
        super().__init__()
        self.use_fm, self.use_dnn, self.output_prob = fm, dnn, output_prob
        self.linear = FieldEmbedding(field_dims, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        if fm or dnn:
            self.embed = FieldEmbedding(field_dims, embed_dim)
        if dnn:
            self.deep = mlp(len(field_dims) * embed_dim, hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.bias + self.linear(x).sum(dim=(1, 2))
        if self.use_fm or self.use_dnn:
            e = self.embed(x)
            if self.use_fm:
                sq_of_sum = e.sum(1).pow(2)
                sum_of_sq = e.pow(2).sum(1)
                logit = logit + 0.5 * (sq_of_sum - sum_of_sq).sum(1)
            if self.use_dnn:
                logit = logit + self.deep(e.flatten(1)).squeeze(1)
        return torch.sigmoid(logit) if self.output_prob else logit


ARCHS = {
    "lr": dict(fm=False, dnn=False),
    "fm": dict(fm=True, dnn=False),
    "dnn": dict(fm=False, dnn=True),
    "deepfm": dict(fm=True, dnn=True),
}


class Tower(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden=(256, 128), out_dim=64):
        super().__init__()
        self.embed = FieldEmbedding(field_dims, embed_dim)
        self.net = mlp(len(field_dims) * embed_dim, hidden, dropout=0.0, out=out_dim, bn=False)

    def forward(self, x):
        return nn.functional.normalize(self.net(self.embed(x).flatten(1)), dim=-1)


class TwoTower(nn.Module):
    """Query tower over context fields, item tower over ad fields.

    Scores are a temperature-scaled cosine, so item vectors drop straight into
    a FAISS inner-product index.
    """

    def __init__(self, q_dims, i_dims, out_dim=64):
        super().__init__()
        self.q = Tower(q_dims, out_dim=out_dim)
        self.i = Tower(i_dims, out_dim=out_dim)
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def score(self, qv, iv):
        return self.log_scale.exp() * qv @ iv.T
