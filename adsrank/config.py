"""Shared paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ADSRANK_DATA", Path.home() / "adsrank_data"))
RAW = DATA / "raw"
PROC = DATA / "processed"
ART = DATA / "artifacts"
RESULTS = REPO / "results"
SITE_ASSETS = REPO / "docs" / "assets"

for _p in (RAW, PROC, ART, RESULTS, SITE_ASSETS):
    _p.mkdir(parents=True, exist_ok=True)

N_INT = 13
N_CAT = 26
INT_COLS = [f"integer_feature_{i}" for i in range(1, N_INT + 1)]
CAT_COLS = [f"categorical_feature_{i}" for i in range(1, N_CAT + 1)]
# Short names used everywhere downstream: I1..I13, C1..C26 (Kaggle convention).
FIELDS = [f"I{i}" for i in range(1, N_INT + 1)] + [f"C{i}" for i in range(1, N_CAT + 1)]
N_FIELDS = len(FIELDS)

SEED = 20261011
