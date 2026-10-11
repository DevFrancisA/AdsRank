"""Assemble everything the datasheet site needs into docs/assets/.

  demo.json            evaluation-board payload (queries, catalog, field maps)
  ranker_web.onnx      DeepFM with its embedding table pruned to demo rows
  query_tower_web.onnx query tower, same pruning
  item_vectors.bin     float32 [N_items, d+1] (scaled item vector ⊕ bias)

Pruning keeps the model exact: rows are copied verbatim and only the per-field
index spaces shrink, so the browser computes the same logits as Python for
every input it can express.
"""
from __future__ import annotations

import json
import math

import numpy as np
import onnxruntime as ort
import torch

from . import data
from .config import ART, FIELDS, N_INT, PROC, SITE_ASSETS
from .models import ARCHS, CTRModel, Tower, TwoTower
from .retrieval import Space, load_ranker


def _rows_for(local_vals, dims):
    """Global table rows for per-field local value lists."""
    offs = np.concatenate([[0], np.cumsum(dims)[:-1]])
    return np.concatenate([offs[j] + np.asarray(v) for j, v in enumerate(local_vals)])


def prune_embedding(src, local_vals, dims, out_dim):
    rows = torch.from_numpy(_rows_for(local_vals, dims))
    from .models import FieldEmbedding
    fe = FieldEmbedding([len(v) for v in local_vals], out_dim)
    fe.emb.weight.data.copy_(src.emb.weight.data[rows])
    return fe


def int_token(b: int) -> str:
    if b == 1:
        return "∅"
    if 2 <= b <= 5:
        return "≤ −1" if b == 2 else str(b - 3)
    lo, hi = math.exp(math.sqrt(b - 6)), math.exp(math.sqrt(b - 5))
    lo, hi = math.ceil(lo), math.ceil(hi) - 1
    return str(lo) if lo >= hi else f"{lo}–{hi}"


def cat_token(field, v, vocab) -> str:
    return "OOV" if v == 0 else "∅" if v == 1 else vocab[field][v - 2]


def build_demo(sp: Space, vocab):
    ranker = load_ranker()
    ck = torch.load(ART / "two_tower_distilled.pt")
    tt = TwoTower(sp.ctx_dims, sp.item_dims)
    tt.load_state_dict(ck["tt"])
    tt.eval()
    items = np.load(ART / "catalog_items.npy")
    iv = np.load(ART / "item_vectors.npy")
    explorer = json.loads((ART / "explorer.json").read_text())
    meta = data.meta()
    dims = meta["field_dims"]

    # per-field set of local values the board can address
    used = [set([0, 1]) for _ in FIELDS]
    for q in explorer:
        for c, j in enumerate(sp.ctx_idx):
            used[j].add(int(q["q"][c]))
    for c, j in enumerate(sp.item_idx):
        used[j].update(int(v) for v in items[:, c])
    vals = [np.array(sorted(u)) for u in used]
    remap = [{int(v): r for r, v in enumerate(vs)} for vs in vals]

    # pruned ranker (exact copy of the rows it keeps)
    web = CTRModel([len(v) for v in vals], embed_dim=ranker.embed.emb.embedding_dim, output_prob=True, **ARCHS["deepfm"])
    web.load_state_dict({k: v for k, v in ranker.state_dict().items() if not k.startswith(("linear.emb", "embed.emb"))}, strict=False)
    web.linear = prune_embedding(ranker.linear, vals, dims, 1)
    web.embed = prune_embedding(ranker.embed, vals, dims, ranker.embed.emb.embedding_dim)
    web.eval()

    ctx_vals = [vals[j] for j in sp.ctx_idx]
    qt = Tower([len(v) for v in ctx_vals], out_dim=tt.q.net[-1].out_features)
    qt.net.load_state_dict(tt.q.net.state_dict())
    qt.embed = prune_embedding(tt.q.embed, ctx_vals, sp.ctx_dims, tt.q.embed.emb.embedding_dim)
    qt.eval()

    def enc(j, v):
        return remap[j][int(v)]

    queries = []
    for qi, q in enumerate(explorer):
        ctx = [enc(j, v) for j, v in zip(sp.ctx_idx, q["q"])]
        toks = [int_token(int(v)) if j < N_INT else cat_token(FIELDS[j], int(v), vocab) for j, v in zip(sp.ctx_idx, q["q"])]
        queries.append({"id": qi, "ctx": ctx, "tokens": toks,
                        "oracle": [o["item"] for o in q["oracle"]], "funnel": [o["item"] for o in q["funnel"]]})
    item_enc = np.array([[enc(j, v) for j, v in zip(sp.item_idx, row)] for row in items], dtype=np.int64)

    # verify pruning is exact on the explorer queries
    full = np.empty((len(items), len(FIELDS)), dtype=np.int64)
    pr = np.empty_like(full)
    q0 = np.asarray(explorer[0]["q"])
    full[:, sp.ctx_idx] = q0
    full[:, sp.item_idx] = items
    pr[:, sp.ctx_idx] = queries[0]["ctx"]
    pr[:, sp.item_idx] = item_enc
    with torch.no_grad():
        a = torch.sigmoid(ranker(torch.from_numpy(full))).numpy()
        b = web(torch.from_numpy(pr)).numpy()
        qa = tt.q(torch.from_numpy(np.asarray([e["q"] for e in explorer[:8]], dtype=np.int64))).numpy()
        qb = qt(torch.from_numpy(np.asarray([e["ctx"] for e in queries[:8]], dtype=np.int64))).numpy()
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()
    assert np.abs(qa - qb).max() < 1e-5, np.abs(qa - qb).max()

    for model, name, F in ((web, "ranker_web.onnx", len(FIELDS)), (qt, "query_tower_web.onnx", len(sp.ctx_idx))):
        torch.onnx.export(model, (torch.zeros(4, F, dtype=torch.int64),), str(SITE_ASSETS / name),
                          input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                          dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
    sess = ort.InferenceSession(str(SITE_ASSETS / "ranker_web.onnx"))
    assert np.abs(sess.run(None, {"x": pr})[0] - b).max() < 1e-5

    iv.astype("<f4").tofile(SITE_ASSETS / "item_vectors.bin")
    key_field = FIELDS[sp.item_idx[0]]
    demo = {
        "fields": FIELDS, "ctx_idx": sp.ctx_idx.tolist(), "item_idx": sp.item_idx.tolist(),
        "item_fields": [FIELDS[j] for j in sp.item_idx], "key_field": key_field,
        "missing": [remap[j][1] for j in range(len(FIELDS))],
        "n_items": int(len(items)), "vec_dim": int(iv.shape[1]),
        "items": item_enc.reshape(-1).tolist(),
        "item_keys": [vocab[key_field][int(k) - 2] for k in items[:, 0]],
        "queries": queries, "k_default": int(round(0.1 * len(items))),
    }
    (SITE_ASSETS / "demo.json").write_text(json.dumps(demo, separators=(",", ":")))
    print("  demo assets written to", SITE_ASSETS)


def main():
    sp = Space()
    vocab = json.loads((PROC / "vocab.json").read_text())
    build_demo(sp, vocab)


if __name__ == "__main__":
    main()
