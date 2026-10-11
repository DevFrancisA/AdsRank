"""Assemble everything the datasheet site needs into docs/assets/.

  report.json          every number the page prints
  demo.json            evaluation-board payload (queries, catalog, field maps)
  ranker_web.onnx      DeepFM with its embedding table pruned to demo rows
  query_tower_web.onnx query tower, same pruning
  item_vectors.bin     float32 [N_items, d+1] (scaled item vector ⊕ bias)

Pruning keeps the model exact: rows are copied verbatim and only the per-field
index spaces shrink, so the browser computes the same logits as Python for
every input it can express.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import platform
import subprocess

import faiss
import numpy as np
import onnxruntime as ort
import torch

from . import data
from .config import ART, FIELDS, N_INT, PROC, RESULTS, SITE_ASSETS
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
    freq = np.load(ART / "catalog_freq.npy")
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
        "item_freq": freq.tolist(),
        "queries": queries, "k_default": int(round(0.1 * len(items))),
    }
    (SITE_ASSETS / "demo.json").write_text(json.dumps(demo, separators=(",", ":")))
    print(f"  demo: {len(queries)} queries, pruned rows {sum(len(v) for v in vals):,} of {sum(dims):,}")


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def os_name() -> str:
    try:
        rel = dict(l.strip().split("=", 1) for l in open("/etc/os-release") if "=" in l)
        name = rel.get("PRETTY_NAME", "").strip('"')
        if "microsoft" in platform.release().lower():
            name += " (WSL2)"
        return name
    except OSError:
        return platform.platform()


def build_report(sp: Space, vocab):
    meta = data.meta()
    split = json.loads((RESULTS / "field_split.json").read_text())
    runs, seeds, sweeps = {}, {}, {}
    for a in ARCHS:
        runs[a] = json.loads((RESULTS / "runs" / f"{a}.json").read_text())
        p = RESULTS / "runs" / f"{a}_l2_sweep.json"
        if p.exists():
            sweeps[a] = json.loads(p.read_text())
        seeds[a] = [{"seed": 1, **runs[a]["test"]}] + [
            {"seed": int(f.stem.split("_s")[-1]), **json.loads(f.read_text())["test"]}
            for f in sorted((RESULTS / "runs").glob(f"{a}_s*.json"))]
    ret = json.loads((RESULTS / "retrieval.json").read_text())
    exp = json.loads((RESULTS / "export.json").read_text())
    lat_py = json.loads((RESULTS / "latency_py.json").read_text())
    lat_cpp = [json.loads(l) for l in (RESULTS / "latency_cpp.jsonl").read_text().splitlines() if l.strip()]

    Xte, _ = data.load("test")
    oov = (np.asarray(Xte[:200_000]) == 0).mean(axis=0)
    fields = []
    for j, name in enumerate(FIELDS):
        side = "item" if j in sp.item_idx else "context"
        if j < N_INT:
            desc, o = "Count feature; ≤2 verbatim, else ⌊ln²x⌋ bucket", None
        else:
            desc, o = "Hashed categorical (32-bit, anonymised)", float(oov[j])
        if name == split["key"]:
            desc = "Hashed categorical — ad key: determines " + ", ".join(split["item_fields"][1:])
        elif side == "item":
            desc = f"Hashed categorical — ad attribute (U | {split['key']} < {split['threshold']})"
        fields.append({"name": name, "kind": "count" if j < N_INT else "cat", "side": side,
                       "card": int(meta["field_dims"][j]), "oov": o, "desc": desc})
    fields_out = fields
    fields_out_pin40 = {"name": "pCTR", "desc": "Output: sigmoid click probability, float32"}

    best = min((r for r in lat_cpp if r["mode"] == "tuned"), key=lambda r: r["p95"])
    eager = lat_py["pytorch_eager"]
    h = ret["headline"]["two_tower_distilled"]
    hc = ret["headline"]["two_tower_clicks"]
    sw = sweeps.get("deepfm", [])
    sw0 = next((r for r in sw if r["l2"] == 0), None)
    revisions = [
        {"change": "Data", "desc": f"{meta['n_rows']:,} impressions from the Criteo 1TB click logs (Hugging Face mirror), random "
                                   f"80/10/10 split; base CTR {100 * meta['ctr']['test']:.2f}% on test."},
        {"change": "Ranker", "desc": "DeepFM and LR / FM / DNN ablations peak at the end of epoch 1 and overfit in epoch 2. "
                                     "Added per-configuration embedding-L2 selection on validation and LR decay on plateau"
                                     + (f" (DeepFM val log loss {sw0['val']['logloss']:.5f} → {runs['deepfm']['val']['logloss']:.5f})." if sw0 else ".")},
        *([{"change": "Ad entity", "desc": (lambda pr: f"The entropy test alone picked {split.get('entropy_key', split['key'])} as the ad key, but its catalog "
              f"ranking was nearly query-independent (one global 1% list retained {100 * next(c for c in pr['candidates'] if c['key'] == split.get('entropy_key'))['prior_ret_1pct']:.1f}% of top-10s). "
              f"Added a probe over all candidate entities and switched to the most query-dependent, {pr['selected']} "
              f"({100 * next(c for c in pr['candidates'] if c['key'] == pr['selected'])['prior_ret_1pct']:.1f}%).")(json.loads((RESULTS / 'split_probe.json').read_text()))}]
          if (RESULTS / "split_probe.json").exists() and split.get("entropy_key") not in (None, split["key"]) else []),
        {"change": "Retrieval", "desc": f"Click-trained two-tower retained {100 * hc['retention']:.1f}% of the oracle top-10 at K = {hc['k']}. "
                                        f"Replaced the objective with listwise distillation from DeepFM plus hard negatives: {100 * h['retention']:.1f}%."},
        {"change": "Serving", "desc": f"Exported to ONNX (max |Δp| {exp['parity']['max_abs_diff']:.1e}); C++ harness with thread sweep and IoBinding "
                                      f"buffers: p95 {eager['p95']:.2f} ms (PyTorch eager) → {best['p95']:.2f} ms ({best['threads']} threads)."},
        {"change": "Rev A", "desc": f"Initial release of this sheet, {dt.date.today():%B %Y}."},
    ]
    vers = {"torch": torch.__version__.split("+")[0], "faiss": faiss.__version__, "ort": ort.__version__}
    operating = [
        ["Dataset", "Criteo 1TB click logs, day 2015-02-15 shards", f"{meta['n_rows']:,} impressions"],
        ["Split", "random, seeded; train / val / test", " / ".join(f"{v / 1e6:.1f}M" for v in meta["split_sizes"].values())],
        ["Base CTR", "test split", f"{100 * meta['ctr']['test']:.2f}%"],
        ["Embedding table", f"39 fields, min count {meta['min_count']}", f"{sum(meta['field_dims']):,} rows × 16"],
        ["Catalog", f"most frequent {split['key']} ad keys", f"{ret['n_items']:,} items"],
        ["Evaluation queries", "held-out test contexts", f"{ret['n_eval_queries']:,}"],
        ["Request shape", "one context × K candidates", f"int64[{exp['batch']}, 39]"],
        ["CPU", cpu_name(), f"{torch.get_num_threads()} threads"],
        ["OS", os_name(), "x86-64"],
        ["Software", "PyTorch / FAISS / ONNX Runtime", f"{vers['torch']} / {vers['faiss']} / {vers['ort']}"],
    ]

    def sz(p):
        return p.stat().st_size if p.exists() else 0
    artifacts = [
        {"name": "deepfm.onnx", "format": "ONNX opset 18", "bytes": sz(ART / "deepfm.onnx"), "devinfo": "ARK-3 ranker",
         "contents": "DeepFM, int64[B,39] → float32[B]; sigmoid in-graph"},
        {"name": "two_tower_distilled.pt", "format": "PyTorch state dict", "bytes": sz(ART / "two_tower_distilled.pt"), "devinfo": "ARK-3 retriever",
         "contents": "query + item towers, per-item bias"},
        {"name": "item_vectors.bin", "format": f"float32 [{ret['n_items']}, {ret['faiss']['dim']}]", "bytes": sz(SITE_ASSETS / "item_vectors.bin"),
         "contents": "scaled item vectors ⊕ bias, FAISS input"},
        {"name": "ranker_web.onnx", "format": "ONNX opset 17", "bytes": sz(SITE_ASSETS / "ranker_web.onnx"), "devinfo": "ARK-3 web (pruned)",
         "contents": "DeepFM with embedding rows pruned to evaluation-board vocabulary"},
        {"name": "query_tower_web.onnx", "format": "ONNX opset 17", "bytes": sz(SITE_ASSETS / "query_tower_web.onnx"),
         "contents": "query tower, same pruning"},
        {"name": "requests.bin", "format": "int64 replay file", "bytes": sz(ART / "requests.bin"),
         "contents": f"{exp['n_requests']} fine-ranking requests for the latency harness"},
    ]
    for a in runs.values():
        a["curve"] = [{k: c[k] for k in ("epoch", "auc", "logloss")} for c in a["curve"]]
    report = {
        "generated": f"{dt.date.today():%Y-%m-%d}", "month": f"{dt.date.today():%B %Y}".upper(),
        "dataset": {"rows": meta["n_rows"], "split_sizes": meta["split_sizes"], "ctr": meta["ctr"],
                    "emb_rows": int(sum(meta["field_dims"])), "min_count": meta["min_count"]},
        "fields": fields_out, "pin40": fields_out_pin40,
        "field_split": {k: split[k] for k in ("key", "item_fields", "context_fields", "threshold", "cat_fields", "U")},
        "ranker": {"runs": runs, "seeds": seeds, "sweeps": sweeps},
        "retrieval": {k: v for k, v in ret.items()},
        "export": exp, "latency": {"py": lat_py, "cpp": lat_cpp},
        "revisions": revisions, "operating": operating, "artifacts": artifacts,
        "probe": json.loads((RESULTS / "split_probe.json").read_text()) if (RESULTS / "split_probe.json").exists() else None,
    }
    (SITE_ASSETS / "report.json").write_text(json.dumps(report, separators=(",", ":")))
    print(f"  report.json written ({(SITE_ASSETS / 'report.json').stat().st_size / 1e3:.0f} kB)")


def main():
    sp = Space()
    vocab = json.loads((PROC / "vocab.json").read_text())
    build_demo(sp, vocab)
    build_report(sp, vocab)


if __name__ == "__main__":
    main()
