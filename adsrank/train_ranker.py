"""Train CTR rankers and run the feature-interaction ablation.

    python -m adsrank.train_ranker --arch lr fm dnn deepfm
    python -m adsrank.train_ranker --arch deepfm --seeds 1 2 3

Every run evaluates on the validation split several times per epoch, keeps
the best checkpoint by validation log loss, and reports held-out test metrics
for that checkpoint only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from . import data
from .config import ART, RESULTS
from .models import ARCHS, CTRModel

torch.set_num_threads(int(os.environ.get("ADSRANK_THREADS", 16)))

HPARAMS = {
    #          lr     epochs  batch
    "lr":     (3e-3, 6,      4096),
    "fm":     (1e-3, 6,      4096),
    "dnn":    (1e-3, 6,      4096),
    "deepfm": (1e-3, 6,      4096),
}
EVALS_PER_EPOCH = 4
PATIENCE = 2        # consecutive non-improving evals before stopping
LR_DECAY = 0.3      # applied to every non-improving eval


@torch.no_grad()
def predict(model, X, batch=65536) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[s:s + batch], dtype=np.int64))
        out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}


def calibration(y, p, bins=12) -> list:
    # quantile bins so every point carries the same mass
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [{"pred": float(p[idx == b].mean()), "obs": float(y[idx == b].mean())} for b in range(bins)]


def selected_l2(arch: str) -> float:
    """Embedding L2 chosen for `arch` by its validation sweep (0 if never swept)."""
    p = RESULTS / "runs" / f"{arch}_l2_sweep.json"
    if not p.exists():
        return 0.0
    return min(json.loads(p.read_text()), key=lambda r: r["val"]["logloss"])["l2"]


def emb_l2(model) -> torch.Tensor:
    reg = model.linear.emb.weight.pow(2).sum()
    if hasattr(model, "embed"):
        reg = reg + model.embed.emb.weight.pow(2).sum()
    return reg


def train(arch: str, seed: int, embed_dim: int = 16, tag: str | None = None, l2: float = 0.0,
          save: bool = True, step_decay: float = 1.0) -> dict:
    tag = tag or (arch if seed == 1 else f"{arch}_s{seed}")
    lr, epochs, bs = HPARAMS[arch]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    meta = data.meta()
    Xtr, ytr = data.load("train")
    Xtr = np.ascontiguousarray(Xtr)
    Xva, yva = data.load("val")
    Xte, yte = data.load("test")

    model = CTRModel(meta["field_dims"], embed_dim=embed_dim, **ARCHS[arch])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()

    steps_per_epoch = len(Xtr) // bs
    eval_every = steps_per_epoch // EVALS_PER_EPOCH
    curve, best, best_state, bad, step = [], (np.inf, None), None, 0, 0
    t0 = time.time()
    print(f"[{tag}] {n_params:,} params, {steps_per_epoch} steps/epoch")
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        model.train()
        for b in range(steps_per_epoch):
            idx = np.sort(perm[b * bs:(b + 1) * bs])
            xb = torch.from_numpy(Xtr[idx].astype(np.int64))
            yb = torch.from_numpy(ytr[idx])
            loss = lossf(model(xb), yb)
            if l2:
                loss = loss + l2 * emb_l2(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                for g in opt.param_groups:
                    g["lr"] *= step_decay
                m = metrics(yva, predict(model, Xva))
                m.update(epoch=round(step / steps_per_epoch, 3), train_loss=float(loss), t=round(time.time() - t0, 1))
                curve.append(m)
                print(f"[{tag}] ep {m['epoch']:.2f}  val auc {m['auc']:.4f}  logloss {m['logloss']:.4f}  ({m['t']}s)")
                if m["logloss"] < best[0] - 1e-5:
                    best = (m["logloss"], m["epoch"])
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    for g in opt.param_groups:
                        g["lr"] *= LR_DECAY
                if bad >= PATIENCE:
                    break
                model.train()
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    pte = predict(model, Xte)
    res = {
        "tag": tag, "arch": arch, "seed": seed, "embed_dim": embed_dim, "params": n_params, "l2": l2, "step_decay": step_decay,
        "lr": lr, "batch": bs, "best_epoch": best[1], "train_seconds": round(time.time() - t0, 1),
        "val": metrics(yva, predict(model, Xva)), "test": metrics(yte, pte), "curve": curve,
    }
    if arch == "deepfm" and seed == 1:
        res["calibration"] = calibration(yte, pte)
    if not save:
        print(f"[{tag}] VAL auc {res['val']['auc']:.4f}  logloss {res['val']['logloss']:.5f}")
        return res
    torch.save({"state": model.state_dict(), "arch": arch, "embed_dim": embed_dim,
                "field_dims": meta["field_dims"]}, ART / f"{tag}.pt")
    (RESULTS / "runs").mkdir(exist_ok=True)
    (RESULTS / "runs" / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{tag}] TEST auc {res['test']['auc']:.4f}  logloss {res['test']['logloss']:.5f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=list(ARCHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--tag")
    ap.add_argument("--l2", type=float, nargs="+", default=None,
                    help="embedding L2; several values = validation sweep (best is kept); "
                         "omitted = the value the sweep selected")
    a = ap.parse_args()
    for arch in a.arch:
        l2 = a.l2 or [selected_l2(arch)]
        if len(l2) > 1:
            sweep = []
            for v in l2:
                r = train(arch, 1, a.embed_dim, f"{arch}_l2_{v:g}", l2=v)
                sweep.append({"l2": v, "val": r["val"], "test": r["test"], "best_epoch": r["best_epoch"]})
            (RESULTS / "runs" / f"{arch}_l2_sweep.json").write_text(json.dumps(sweep, indent=2))
            best = min(sweep, key=lambda r: r["val"]["logloss"])["l2"]
            print(f"[{arch}] selected l2={best:g} on validation log loss")
            # the seed-1 sweep run *is* the selected model: promote it instead of refitting
            src = f"{arch}_l2_{best:g}"
            (ART / f"{src}.pt").replace(ART / f"{arch}.pt")
            res = json.loads((RESULTS / "runs" / f"{src}.json").read_text())
            res["tag"] = arch
            (RESULTS / "runs" / f"{arch}.json").write_text(json.dumps(res, indent=2))
            for v in l2:
                (RESULTS / "runs" / f"{arch}_l2_{v:g}.json").unlink(missing_ok=True)
                (ART / f"{arch}_l2_{v:g}.pt").unlink(missing_ok=True)
            l2, seeds = [best], [s for s in a.seeds if s != 1]
        else:
            seeds = a.seeds
        for s in seeds:
            train(arch, s, a.embed_dim, a.tag, l2=l2[0])
