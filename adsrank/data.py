"""Download a Criteo sample and turn it into dense integer field indices.

Source: Criteo 1TB Click Logs (Hugging Face mirror, `criteo/CriteoClickLogs`),
which shares its schema with the Kaggle Display Advertising Challenge data:
one label, 13 count features, 26 hashed categorical features.

Encoding (every field becomes a categorical index, so the ranker is a pure
field-embedding model):
  * integer features -> missing / <=2 kept verbatim, larger values bucketed by
    floor(ln(x)^2), the discretisation used by the Kaggle-winning solutions.
  * categorical features -> vocabulary fit on the *train split only*; values
    seen fewer than MIN_COUNT times share an OOV slot, missing has its own slot.

Index 0 of every field is OOV, index 1 is "missing".
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import CAT_COLS, FIELDS, INT_COLS, PROC, RAW, SEED

HF_REPO = "criteo/CriteoClickLogs"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data/day={{day}}"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{{path}}"
MIN_COUNT = 10
OOV, MISSING = 0, 1


def download(n_rows: int, day: str = "2015-02-15") -> list:
    with urllib.request.urlopen(HF_API.format(day=day)) as r:
        listing = sorted(
            (f for f in json.load(r) if f["path"].endswith(".parquet")),
            key=lambda f: f["path"],
        )
    have, files = 0, []
    for f in listing:
        dst = RAW / f["path"].rsplit("/", 1)[-1]
        if not dst.exists():
            print(f"  fetching {dst.name} ({f['size'] / 1e6:.0f} MB)")
            urllib.request.urlretrieve(HF_FILE.format(path=f["path"]), dst)
        files.append(dst)
        have += pq.ParquetFile(dst).metadata.num_rows
        if have >= n_rows:
            break
    print(f"  {have:,} rows across {len(files)} shards")
    return files


def _bucket_int(col: pd.Series) -> np.ndarray:
    x = col.to_numpy(dtype=np.float64, na_value=np.nan)
    out = np.full(x.shape, MISSING, dtype=np.int64)
    ok = ~np.isnan(x)
    v = x[ok]
    # negative counts (only I2 has them) collapse to a single bucket
    b = np.where(v <= 2, np.maximum(v, -1) + 3, np.floor(np.log(np.maximum(v, 1)) ** 2) + 6)
    out[ok] = b.astype(np.int64)
    return out


def preprocess(n_rows: int) -> None:
    files = sorted(RAW.glob("*.parquet"))
    tables, have = [], 0
    for f in files:
        t = pq.read_table(f).to_pandas()
        tables.append(t)
        have += len(t)
        if have >= n_rows:
            break
    df = pd.concat(tables, ignore_index=True).iloc[:n_rows]
    assert len(df) == n_rows, f"only {len(df):,} rows on disk"
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_rows)
    n_tr, n_va = int(0.8 * n_rows), int(0.1 * n_rows)
    split = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    train_mask = np.zeros(n_rows, bool)
    train_mask[split["train"]] = True

    X = np.zeros((n_rows, len(FIELDS)), dtype=np.int32)
    vocabs = {}
    for j, c in enumerate(INT_COLS):
        X[:, j] = _bucket_int(df[c])
    # Integer buckets are a closed set; size them from the full sample's max.
    int_dims = (X[:, : len(INT_COLS)].max(axis=0) + 1).tolist()

    cat_dims = []
    for k, c in enumerate(CAT_COLS):
        j = len(INT_COLS) + k
        s = df[c]
        counts = s[train_mask].value_counts()
        keep = counts[counts >= MIN_COUNT].index
        mapping = pd.Series(np.arange(2, 2 + len(keep)), index=keep)
        enc = s.map(mapping)
        col = np.where(s.isna(), MISSING, enc.fillna(OOV)).astype(np.int32)
        X[:, j] = col
        cat_dims.append(len(keep) + 2)
        vocabs[FIELDS[j]] = keep.tolist()
        print(f"  {FIELDS[j]:>4}: {len(keep):>7,} values kept  (oov rate {np.mean(col == OOV):.3%})")

    y = df["label"].to_numpy(np.float32)
    for name, idx in split.items():
        np.save(PROC / f"X_{name}.npy", X[idx])
        np.save(PROC / f"y_{name}.npy", y[idx])
    meta = {
        "fields": FIELDS,
        "field_dims": int_dims + cat_dims,
        "n_rows": n_rows,
        "split_sizes": {k: int(len(v)) for k, v in split.items()},
        "ctr": {k: float(y[v].mean()) for k, v in split.items()},
        "min_count": MIN_COUNT,
        "source": f"huggingface.co/datasets/{HF_REPO} day {files[0].name if files else ''}",
    }
    (PROC / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(PROC / "vocab.json", "w") as f:
        json.dump(vocabs, f)
    print(json.dumps({k: meta[k] for k in ("split_sizes", "ctr")}, indent=2))
    print(f"  total embedding rows: {sum(meta['field_dims']):,}")


def load(split: str):
    X = np.load(PROC / f"X_{split}.npy", mmap_mode="r")
    y = np.load(PROC / f"y_{split}.npy")
    return X, y


def meta() -> dict:
    return json.loads((PROC / "meta.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        download(a.rows)
    preprocess(a.rows)
