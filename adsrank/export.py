"""Export the DeepFM ranker to ONNX and build the latency benchmark requests.

A *request* is one query context against K retrieved candidates - exactly the
fine-ranking stage of the funnel - so every latency number in the repo is
measured on the tensor shape production would see.

Outputs (in $ADSRANK_DATA/artifacts):
  deepfm.onnx           int64[B, 39] -> float32[B] click probabilities
  requests.bin          header int64 (n_req, B, F) + int64 payload
"""
from __future__ import annotations

import json
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from . import data
from .config import ART, RESULTS
from .models import CTRModel
from .retrieval import HEADLINE_FRAC, Space

N_REQ = 256


def export(tag="deepfm", path=None) -> str:
    ck = torch.load(ART / f"{tag}.pt")
    from .models import ARCHS
    m = CTRModel(ck["field_dims"], embed_dim=ck["embed_dim"], output_prob=True, **ARCHS[ck["arch"]])
    m.load_state_dict(ck["state"])
    m.eval()
    path = str(path or ART / f"{tag}.onnx")
    x = torch.zeros(8, len(ck["field_dims"]), dtype=torch.int64)
    torch.onnx.export(
        m, (x,), path, input_names=["x"], output_names=["p"], opset_version=18,
        dynamo=False, dynamic_axes={"x": {0: "batch"}, "p": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(path))
    return path, m


def build_requests(B: int) -> np.ndarray:
    sp = Space()
    items = np.load(ART / "catalog_items.npy")
    q = np.load(ART / "eval_queries.npy")
    explorer = json.loads((ART / "explorer.json").read_text())
    reqs = np.empty((N_REQ, B, len(data.meta()["fields"])), dtype=np.int64)
    rng = np.random.default_rng(0)
    for r in range(N_REQ):
        cand = np.array(explorer[r]["candidates"]) if r < len(explorer) else rng.choice(len(items), B, replace=False)
        reqs[r] = sp.compose(q[r:r + 1], items[cand[:B]])
    return reqs


def main():
    path, model = export()
    items = np.load(ART / "catalog_items.npy")
    B = int(round(HEADLINE_FRAC * len(items)))
    reqs = build_requests(B)
    with open(ART / "requests.bin", "wb") as f:
        np.array(reqs.shape, dtype=np.int64).tofile(f)
        reqs.tofile(f)

    # parity: torch vs onnxruntime on real held-out rows
    Xte, _ = data.load("test")
    xs = np.asarray(Xte[:20000], dtype=np.int64)
    with torch.no_grad():
        pt = model(torch.from_numpy(xs)).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    po = sess.run(None, {"x": xs})[0]
    parity = {"rows": len(xs), "max_abs_diff": float(np.abs(pt - po).max()),
              "mean_abs_diff": float(np.abs(pt - po).mean())}
    size = (ART / "deepfm.onnx").stat().st_size
    out = {"onnx_bytes": size, "batch": B, "n_requests": N_REQ, "parity": parity,
           "opset": 18, "ort_version": ort.__version__, "torch_version": torch.__version__}
    (RESULTS / "export.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
