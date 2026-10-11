"""Which inferred entity makes a meaningful retrieval benchmark?

fields.py finds several clusters of functionally dependent fields. Any of
them could be "the ad". For retrieval to be worth benchmarking, the ranker's
ordering of catalog items has to depend on the query - otherwise one global
shortlist does the job. For each candidate key this probe builds the
catalog, scores held-out queries against it with the trained DeepFM, and
reports

  distinct_top10   distinct items across all queries' top-10 / (10 * queries)
  prior_ret@1%     retention@10 of ONE global shortlist (mean DeepFM logit
                   over training contexts) holding 1% of the catalog

Lower prior retention = more personalised = harder, more honest benchmark.
The key with the most query-dependent ranking becomes the item side.
"""
from __future__ import annotations

import json

import numpy as np

from . import data
from .config import FIELDS, RESULTS, SEED
from .fields import DETERMINED

N_Q, N_PRIOR, N_ITEMS = 200, 256, 10_000


class ProbeSpace:
    def __init__(self, item_idx):
        self.item_idx = np.array(item_idx)
        self.ctx_idx = np.array([j for j in range(len(FIELDS)) if j not in item_idx])
        self.key = self.item_idx[0]

    def compose(self, q, items):
        from .retrieval import Space
        return Space.compose(self, q, items)


def main():
    from . import retrieval as R
    split = json.loads((RESULTS / "field_split.json").read_text())
    U, cats = np.array(split["U"]), split["cat_fields"]
    card = dict(zip(cats, split["cardinality"]))
    rng = np.random.default_rng(SEED)
    ranker = R.load_ranker()
    Xtr, _ = data.load("train")
    Xtr = np.asarray(Xtr)
    Xte, _ = data.load("test")
    test_rows = np.asarray(Xte[rng.choice(len(Xte), N_Q, replace=False)], dtype=np.int64)
    prior_rows = np.asarray(Xtr[rng.choice(len(Xtr), N_PRIOR, replace=False)], dtype=np.int64)

    out = []
    for cand in split["candidates"]:
        k = cand["field"]
        a = cats.index(k)
        det = [cats[b] for b in range(len(cats)) if b != a and U[a, b] < DETERMINED and card[cats[b]] > 2]
        item_idx = [FIELDS.index(k)] + [FIELDS.index(f) for f in det]
        sp = ProbeSpace(item_idx)
        R.N_ITEMS = min(N_ITEMS, card[k] - 2)
        items, _ = R.build_catalog(sp, Xtr)
        q = test_rows[:, sp.ctx_idx]
        S = np.concatenate([R.teacher_scores(ranker, sp, q[s:s + 25], items) for s in range(0, N_Q, 25)])
        top = np.argsort(-S, 1)[:, :10]
        prior = np.concatenate([R.teacher_scores(ranker, sp, prior_rows[s:s + 32][:, sp.ctx_idx], items)
                                for s in range(0, N_PRIOR, 32)]).mean(0)
        kk = max(10, int(0.01 * len(items)))
        shortlist = set(np.argsort(-prior)[:kk].tolist())
        prior_ret = float(np.mean([len(shortlist & set(t.tolist())) / 10 for t in top]))
        row = {"key": k, "item_fields": [k] + det, "n_items": int(len(items)),
               "distinct_top10": round(len(np.unique(top)) / top.size, 4),
               "prior_ret_1pct": round(prior_ret, 4),
               "logit_spread": round(float(np.mean(S.std(1))), 4)}
        out.append(row)
        print(row)
    best = min(out, key=lambda r: r["prior_ret_1pct"])
    (RESULTS / "split_probe.json").write_text(json.dumps({"candidates": out, "selected": best["key"]}, indent=1))
    # promote the selected entity to the item side used downstream
    split.setdefault("entropy_key", split["key"])
    item = [FIELDS.index(f) for f in best["item_fields"]]
    split.update(key=best["key"], item_fields=best["item_fields"], item_idx=item,
                 context_idx=[j for j in range(len(FIELDS)) if j not in item],
                 context_fields=[f for j, f in enumerate(FIELDS) if j not in item], selected_by="split_probe")
    (RESULTS / "field_split.json").write_text(json.dumps(split, indent=1))
    print("most query-dependent ranking:", best["key"], "-> item fields", best["item_fields"])


if __name__ == "__main__":
    main()
