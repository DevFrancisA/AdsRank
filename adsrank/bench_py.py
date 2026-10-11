"""Python-side latency baselines on the same requests the C++ harness replays.

  pytorch_eager   the trained nn.Module under torch.no_grad (torch's default threads)
  ort_python      onnxruntime InferenceSession with default options
"""
from __future__ import annotations

import json
import time

import numpy as np
import onnxruntime as ort
import torch

from .config import ART, RESULTS
from .models import ARCHS, CTRModel

WARMUP, ITERS = 100, 2000


def load_requests():
    with open(ART / "requests.bin", "rb") as f:
        shape = np.fromfile(f, dtype=np.int64, count=3)
        return np.fromfile(f, dtype=np.int64).reshape(shape)


def stats(lat_ms) -> dict:
    a = np.asarray(lat_ms)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "n": len(a)}


def time_loop(fn, reqs):
    for i in range(WARMUP):
        fn(reqs[i % len(reqs)])
    lat = []
    for i in range(ITERS):
        r = reqs[i % len(reqs)]
        t = time.perf_counter_ns()
        fn(r)
        lat.append((time.perf_counter_ns() - t) / 1e6)
    return lat


def main():
    reqs = load_requests()
    ck = torch.load(ART / "deepfm.pt")
    m = CTRModel(ck["field_dims"], embed_dim=ck["embed_dim"], output_prob=True, **ARCHS[ck["arch"]])
    m.load_state_dict(ck["state"])
    m.eval()

    def torch_fn(r):
        with torch.no_grad():
            return m(torch.from_numpy(r)).numpy()

    sess = ort.InferenceSession(str(ART / "deepfm.onnx"), providers=["CPUExecutionProvider"])

    def ort_fn(r):
        return sess.run(None, {"x": r})[0]

    out = {"batch": int(reqs.shape[1]), "torch_threads": torch.get_num_threads(),
           "pytorch_eager": stats(time_loop(torch_fn, reqs)),
           "ort_python": stats(time_loop(ort_fn, reqs))}
    (RESULTS / "latency_py.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
