"""Infer which Criteo fields describe the *ad* and which describe the *context*.

Criteo does not disclose field semantics, but a retrieval stack needs an item
side. Attributes of the same entity are functionally dependent: an ad id pins
down its advertiser, campaign, and category. So we compute the normalised
conditional entropy U(J|K) = H(J|K) / H(J) for every pair of categorical
fields and pick as the item key the high-cardinality field that (almost)
determines the most other fields. Those fields plus the key form the item;
everything else - including all count features - is query context.

Several clusters usually qualify; split_probe.py then picks, among the top
candidates found here, the entity whose catalog ranking depends most on the
query and overwrites the split with it.
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .config import FIELDS, N_INT, RESULTS

SAMPLE = 1_000_000
DETERMINED = 0.10      # U(J|K) below this means K (almost) fixes J
MIN_KEY_CARD = 2_000


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def cond_entropy(k: np.ndarray, j: np.ndarray) -> float:
    pair = k.astype(np.int64) * (int(j.max()) + 1) + j
    return entropy(np.unique(pair, return_counts=True)[1]) - entropy(np.unique(k, return_counts=True)[1])


def main():
    X, _ = data.load("train")
    X = np.asarray(X[:SAMPLE])
    cats = list(range(N_INT, len(FIELDS)))
    H = {j: entropy(np.unique(X[:, j], return_counts=True)[1]) for j in cats}
    card = {j: int(len(np.unique(X[:, j]))) for j in cats}
    U = np.ones((len(cats), len(cats)))
    for a, k in enumerate(cats):
        for b, j in enumerate(cats):
            if a != b and H[j] > 0:
                U[a, b] = cond_entropy(X[:, k], X[:, j]) / H[j]
            elif a == b:
                U[a, b] = 0.0

    def determined_by(a):
        return [cats[b] for b in range(len(cats)) if b != a and U[a, b] < DETERMINED and card[cats[b]] > 2]

    scored = sorted(
        ((len(determined_by(a)), card[k], k, a) for a, k in enumerate(cats) if card[k] >= MIN_KEY_CARD),
        reverse=True,
    )
    _, _, key, ka = scored[0]
    item = [key] + determined_by(ka)
    ctx = [j for j in range(len(FIELDS)) if j not in item]
    out = {
        "key": FIELDS[key],
        "item_fields": [FIELDS[j] for j in item],
        "context_fields": [FIELDS[j] for j in ctx],
        "item_idx": item,
        "context_idx": ctx,
        "threshold": DETERMINED,
        "cat_fields": [FIELDS[j] for j in cats],
        "cardinality": [card[j] for j in cats],
        "U": np.round(U, 3).tolist(),
        "candidates": [{"field": FIELDS[k], "determines": n, "card": c} for n, c, k, _ in scored[:6]],
    }
    (RESULTS / "field_split.json").write_text(json.dumps(out, indent=1))
    print("candidates:", out["candidates"])
    print("item key:", out["key"], "| item fields:", out["item_fields"])


if __name__ == "__main__":
    main()
