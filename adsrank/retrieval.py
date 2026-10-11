"""Two-tower retrieval + FAISS candidate recall + DeepFM fine-ranking.

Offline benchmark
-----------------
catalog   the N most frequent item keys in train; each item carries the modal
          value of every item field given its key
queries   context fields of held-out test impressions
oracle    DeepFM scores every (query, item) pair -> exhaustive top-10
funnel    two-tower vectors -> FAISS IVF top-K -> DeepFM scores only those K
          -> top-10
metric    retention@10 = |funnel top-10 ∩ oracle top-10| / 10

Retrievers compared
-------------------
popularity   top-K items by train frequency (same list for every query)
ranker prior top-K items by mean DeepFM logit over 512 training contexts
             (non-personalised, but ranker-aware)
two-tower    trained on clicks, in-batch sampled softmax with logQ correction
distilled    same towers, trained to match DeepFM's listwise item distribution
             (KL over sampled catalog slates)

Item vectors carry a learned bias as an extra coordinate and queries a
constant 1, so FAISS inner product == the two-tower score exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import faiss
import numpy as np
import torch
from torch.nn import functional as F

from . import data
from .config import ART, FIELDS, RESULTS, SEED
from .models import CTRModel, TwoTower

torch.set_num_threads(int(os.environ.get("ADSRANK_THREADS", 16)))

# env overrides exist only for quick smoke tests
N_ITEMS = int(os.environ.get("ADSRANK_N_ITEMS", 10_000))
N_EVAL_Q = int(os.environ.get("ADSRANK_EVAL_Q", 1_000))
TOPN = 10
K_FRACS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
HEADLINE_FRAC = 0.10


def load_ranker(tag="deepfm") -> CTRModel:
    ck = torch.load(ART / f"{tag}.pt")
    from .models import ARCHS
    m = CTRModel(ck["field_dims"], embed_dim=ck["embed_dim"], **ARCHS[ck["arch"]])
    m.load_state_dict(ck["state"])
    return m.eval()


class Space:
    """Field bookkeeping for the context/item split."""

    def __init__(self):
        split = json.loads((RESULTS / "field_split.json").read_text())
        self.item_idx = np.array(split["item_idx"])
        self.ctx_idx = np.array(split["context_idx"])
        self.key = self.item_idx[0]
        dims = np.array(data.meta()["field_dims"])
        self.item_dims, self.ctx_dims = dims[self.item_idx].tolist(), dims[self.ctx_idx].tolist()

    def compose(self, q: np.ndarray, items: np.ndarray) -> np.ndarray:
        """All pairs of q[Bq, Fc] x items[M, Fi] -> x[Bq*M, F]."""
        bq, m = len(q), len(items)
        x = np.empty((bq, m, len(FIELDS)), dtype=np.int64)
        x[:, :, self.ctx_idx] = q[:, None, :]
        x[:, :, self.item_idx] = items[None, :, :]
        return x.reshape(bq * m, -1)


def build_catalog(sp: Space, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    keys = X[:, sp.key]
    vals, counts = np.unique(keys[keys > 1], return_counts=True)  # drop OOV / missing
    order = np.argsort(-counts)[:N_ITEMS]
    top, freq = vals[order], counts[order]
    rows = np.flatnonzero(np.isin(keys, top))
    srt = np.argsort(top)
    df = pd.DataFrame(X[rows][:, sp.item_idx[1:]], columns=list(range(1, len(sp.item_idx))))
    df["r"] = srt[np.searchsorted(top[srt], keys[rows])]
    items = np.zeros((len(top), len(sp.item_idx)), dtype=np.int64)
    items[:, 0] = top
    for c in range(1, len(sp.item_idx)):
        mode = df.groupby(["r", c]).size().groupby(level=0).idxmax()
        items[[r for r, _ in mode], c] = [v for _, v in mode]
    return items, freq


@torch.no_grad()
def teacher_scores(ranker, sp: Space, q: np.ndarray, items: np.ndarray, chunk=262_144) -> np.ndarray:
    x = sp.compose(q, items)
    out = np.empty(len(x), dtype=np.float32)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = ranker(torch.from_numpy(x[s:s + chunk])).numpy()
    return out.reshape(len(q), len(items))


@torch.no_grad()
def teacher_scores_slates(ranker, sp: Space, q: np.ndarray, slates: np.ndarray) -> np.ndarray:
    """Per-query slates: q[Bq, Fc], slates[Bq, M, Fi] -> logits[Bq, M]."""
    bq, m, _ = slates.shape
    x = np.empty((bq, m, len(FIELDS)), dtype=np.int64)
    x[:, :, sp.ctx_idx] = q[:, None, :]
    x[:, :, sp.item_idx] = slates
    return ranker(torch.from_numpy(x.reshape(bq * m, -1))).numpy().reshape(bq, m)


def item_vectors(tt: TwoTower, items: np.ndarray, bias: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        iv = tt.i(torch.from_numpy(items)) * tt.log_scale.exp()
        return torch.cat([iv, bias[:, None]], 1).numpy().astype(np.float32)


def query_vectors(tt: TwoTower, q: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        qv = tt.q(torch.from_numpy(q))
        return torch.cat([qv, torch.ones(len(q), 1)], 1).numpy().astype(np.float32)


def train_distilled(ranker, sp, items, q_pool, steps=4000, bq=256, m_uniform=128, m_hard=64, seed=SEED):
    """Listwise distillation: the student's softmax over a slate matches the teacher's.

    Each query's slate = a shared uniform sample of the catalog (learns the global
    shape) + the student's own current top-m_hard items for that query (hard
    negatives, so it learns to order the head the ranker actually cares about).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tt = TwoTower(sp.ctx_dims, sp.item_dims)
    bias = torch.nn.Parameter(torch.zeros(len(items)))
    opt = torch.optim.Adam([*tt.parameters(), bias], lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    items_t = torch.from_numpy(items)
    warmup, refresh = 400, 100
    log, t0, iv_cache = [], time.time(), None
    for step in range(steps):
        qi = rng.integers(0, len(q_pool), bq)
        q = q_pool[qi]
        qv = tt.q(torch.from_numpy(q))
        uni = np.tile(rng.choice(len(items), m_uniform, replace=False), (bq, 1))
        if step >= warmup:
            if iv_cache is None or step % refresh == 0:
                with torch.no_grad():
                    iv_cache = tt.i(items_t) * tt.log_scale.exp() + 0  # [N, d]
            with torch.no_grad():
                hard = torch.topk(qv @ iv_cache.T + bias, m_hard, dim=1).indices.numpy()
            slate = np.concatenate([uni, hard], 1)
        else:
            slate = uni
        T = torch.from_numpy(teacher_scores_slates(ranker, sp, q, items[slate]))
        iv = tt.i(items_t[torch.from_numpy(slate.reshape(-1))]).view(bq, slate.shape[1], -1)
        S = tt.log_scale.exp() * (qv[:, None, :] * iv).sum(-1) + bias[torch.from_numpy(slate)]
        loss = F.kl_div(F.log_softmax(S, 1), F.log_softmax(T, 1), log_target=True, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            log.append({"step": step, "kl": round(float(loss), 4), "t": round(time.time() - t0, 1)})
            print(f"  [distill] step {step:>5}  kl {float(loss):.4f}  ({time.time() - t0:.0f}s)")
    return tt.eval(), bias.detach(), log


def train_clicks(sp, items, X, y, freq, epochs=4, bs=1024, seed=SEED):
    """Classic retrieval objective: in-batch sampled softmax over clicked impressions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pos_of = {int(k): r for r, k in enumerate(items[:, 0])}
    keys = X[:, sp.key]
    rows = np.flatnonzero((y == 1) & np.isin(keys, items[:, 0]))
    q_all = np.asarray(X[rows][:, sp.ctx_idx], dtype=np.int64)
    it_all = np.array([pos_of[int(k)] for k in keys[rows]])
    logq = torch.from_numpy(np.log(freq / freq.sum()).astype(np.float32))
    tt = TwoTower(sp.ctx_dims, sp.item_dims)
    bias = torch.nn.Parameter(torch.zeros(len(items)))
    opt = torch.optim.Adam([*tt.parameters(), bias], lr=2e-3)
    items_t = torch.from_numpy(items)
    log, t0 = [], time.time()
    for ep in range(epochs):
        perm = rng.permutation(len(rows))
        tot = 0.0
        for b in range(len(perm) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            it = torch.from_numpy(it_all[idx])
            uniq, inv = torch.unique(it, return_inverse=True)
            S = tt.score(tt.q(torch.from_numpy(q_all[idx])), tt.i(items_t[uniq])) + bias[uniq] - logq[uniq]
            loss = F.cross_entropy(S, inv)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        log.append({"epoch": ep + 1, "loss": round(tot / (len(perm) // bs), 4), "t": round(time.time() - t0, 1)})
        print(f"  [clicks] epoch {ep + 1}  loss {log[-1]['loss']}  ({time.time() - t0:.0f}s)")
    return tt.eval(), bias.detach(), log, int(len(rows))


NLIST = 100


def nprobe_for(frac: float) -> int:
    """Probe enough lists to scan ~3x the requested shortlist."""
    return int(min(NLIST, max(8, np.ceil(3 * frac * NLIST))))


def ivf_index(vecs: np.ndarray, nlist=NLIST):
    d = vecs.shape[1]
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(vecs)
    idx.add(vecs)
    return idx


def retention(S_oracle: np.ndarray, cand: np.ndarray, oracle_top: np.ndarray) -> np.ndarray:
    """Per-query overlap of the funnel's top-10 (ranker over cand) with the oracle top-10."""
    sub = np.take_along_axis(S_oracle, np.where(cand < 0, 0, cand), 1)
    sub[cand < 0] = -np.inf
    k = min(TOPN, cand.shape[1])
    pick = np.take_along_axis(cand, np.argpartition(-sub, k - 1, 1)[:, :k], 1)
    return np.array([len(set(a) & set(b)) / TOPN for a, b in zip(pick, oracle_top)])


def main(distill_steps: int):
    rng = np.random.default_rng(SEED)
    sp = Space()
    ranker = load_ranker()
    Xtr, ytr = data.load("train")
    Xtr = np.asarray(Xtr)
    Xte, _ = data.load("test")
    items, freq = build_catalog(sp, Xtr)
    print(f"catalog: {len(items):,} items; head freq {freq[0]:,}, tail freq {freq[-1]:,}")

    q_eval = np.asarray(Xte[rng.choice(len(Xte), N_EVAL_Q, replace=False)][:, sp.ctx_idx], dtype=np.int64)
    q_pool = np.asarray(Xtr[rng.choice(len(Xtr), 400_000, replace=False)][:, sp.ctx_idx], dtype=np.int64)

    t0 = time.time()
    S = np.concatenate([teacher_scores(ranker, sp, q_eval[s:s + 50], items) for s in range(0, N_EVAL_Q, 50)])
    t_oracle = time.time() - t0
    oracle_top = np.argsort(-S, 1)[:, :TOPN]
    print(f"oracle: {S.size:,} ranker evaluations in {t_oracle:.0f}s")

    print("training click two-tower ...")
    tt_c, b_c, log_c, n_clicks = train_clicks(sp, items, Xtr, ytr, freq)
    print("training distilled two-tower ...")
    tt_d, b_d, log_d = train_distilled(ranker, sp, items, q_pool, steps=distill_steps)

    retrievers = {}
    for name, tt, b in (("two_tower_clicks", tt_c, b_c), ("two_tower_distilled", tt_d, b_d)):
        iv, qv = item_vectors(tt, items, b), query_vectors(tt, q_eval)
        retrievers[name] = (iv, qv)

    prior_q = q_pool[rng.choice(len(q_pool), 512, replace=False)]
    prior = np.concatenate([teacher_scores(ranker, sp, prior_q[s:s + 32], items) for s in range(0, 512, 32)]).mean(0)
    prior_order = np.argsort(-prior)

    curves = {"popularity": [], "ranker_prior": [], "two_tower_clicks": [], "two_tower_distilled": []}
    faiss_stats = {}
    for frac in K_FRACS:
        k = int(round(frac * len(items)))
        pop = np.tile(np.arange(k), (N_EVAL_Q, 1))  # catalog is frequency-sorted
        curves["popularity"].append({"frac": frac, "k": k, "retention": float(retention(S, pop, oracle_top).mean())})
        pri = np.tile(prior_order[:k], (N_EVAL_Q, 1))
        curves["ranker_prior"].append({"frac": frac, "k": k, "retention": float(retention(S, pri, oracle_top).mean())})
        for name, (iv, qv) in retrievers.items():
            index = ivf_index(iv)
            exact = np.argsort(-(qv @ iv.T), 1)[:, :k]
            index.nprobe = nprobe_for(frac)
            t1 = time.time()
            _, cand = index.search(qv, k)
            t_search = (time.time() - t1) / N_EVAL_Q * 1e3
            ann_recall = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(cand, exact)]))
            r = retention(S, cand, oracle_top)
            curves[name].append({
                "frac": frac, "k": k, "retention": float(r.mean()),
                "retention_p10": float(np.quantile(r, 0.10)),
                "exact_retention": float(retention(S, exact, oracle_top).mean()),
                "faiss_recall_vs_exact": ann_recall, "faiss_ms_per_query": round(t_search, 3), "nprobe": index.nprobe,
            })
            if abs(frac - HEADLINE_FRAC) < 1e-9:
                faiss_stats[name] = curves[name][-1]
        print(f"K={k:>5} ({frac:.1%})  " + "  ".join(f"{n}={c[-1]['retention']:.3f}" for n, c in curves.items()))

    # Per-query explorer payload for the site (first 48 eval queries, headline K).
    iv, qv = retrievers["two_tower_distilled"]
    k = int(round(HEADLINE_FRAC * len(items)))
    index = ivf_index(iv)
    index.nprobe = nprobe_for(HEADLINE_FRAC)
    _, cand = index.search(qv, k)
    explorer = []
    for qi in range(48):
        c = cand[qi]
        funnel = c[np.argsort(-S[qi, c])[:TOPN]]
        rank_in_retrieval = {int(it): int(np.flatnonzero(c == it)[0]) if it in c else None for it in oracle_top[qi]}
        explorer.append({
            "q": q_eval[qi].tolist(),
            "oracle": [{"item": int(i), "logit": float(S[qi, i]), "retrieval_rank": rank_in_retrieval[int(i)]}
                       for i in oracle_top[qi]],
            "funnel": [{"item": int(i), "logit": float(S[qi, i])} for i in funnel],
            "candidates": c.tolist(),
        })

    out = {
        "n_items": len(items), "n_eval_queries": N_EVAL_Q, "topn": TOPN, "headline_frac": HEADLINE_FRAC,
        "oracle_seconds": round(t_oracle, 1), "oracle_evals": int(S.size),
        "item_fields": [FIELDS[i] for i in sp.item_idx], "context_fields": [FIELDS[i] for i in sp.ctx_idx],
        "curves": curves, "headline": faiss_stats,
        "train_logs": {"two_tower_clicks": log_c, "two_tower_distilled": log_d},
        "n_click_examples": n_clicks, "distill_steps": distill_steps,
        "faiss": {"index": "IVFFlat", "nlist": NLIST, "nprobe": nprobe_for(HEADLINE_FRAC), "dim": int(iv.shape[1])},
    }
    (RESULTS / "retrieval.json").write_text(json.dumps(out, indent=1))
    np.save(ART / "catalog_items.npy", items)
    np.save(ART / "catalog_freq.npy", freq)
    np.save(ART / "item_vectors.npy", iv)
    np.save(ART / "eval_queries.npy", q_eval)
    np.save(ART / "oracle_scores.npy", S)
    (ART / "explorer.json").write_text(json.dumps(explorer))
    torch.save({"tt": tt_d.state_dict(), "bias": b_d}, ART / "two_tower_distilled.pt")
    h = faiss_stats["two_tower_distilled"]
    print(f"\nHEADLINE: K={h['k']} ({HEADLINE_FRAC:.0%} of catalog, {1 - HEADLINE_FRAC:.0%} fewer ranker calls) "
          f"-> retention@10 {h['retention']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill-steps", type=int, default=4000)
    main(ap.parse_args().distill_steps)
