# AdsRank

**A three-stage click-through-rate ranking stack on Criteo — two-tower retrieval, FAISS candidate recall, DeepFM fine-ranking, ONNX Runtime C++ serving.**

**Live site → https://devfrancisa.github.io/AdsRank/** — the full funnel runs in your browser on the evaluation board.

## What it does

An ad request arrives with a context (who, where, when). AdsRank answers it in three stages:

1. **Retrieve** — a query tower embeds the context; a FAISS IVF index returns a shortlist of candidate ads from the catalog.
2. **Rank** — a DeepFM model predicts click probability for every (context, ad) pair in the shortlist.
3. **Serve** — the ranker runs as an ONNX graph inside a C++ ONNX Runtime harness and returns the top ads.

An offline benchmark compares the funnel's output against an oracle in which DeepFM scores the entire catalog, so you can see how small the shortlist can get before ranking quality suffers.

## Features

- **Criteo data pipeline** — downloads a sample of the Criteo click logs, buckets count features with the ⌊ln²x⌋ transform, and fits categorical vocabularies on the training split only.
- **DeepFM ranker** — linear, factorization-machine, and deep MLP terms over one shared field-embedding table, with field offsets baked into the model so the exported graph takes raw field indices.
- **Feature-interaction ablation** — logistic regression, FM, DNN, and DeepFM trained under one protocol: validation-selected embedding L2, learning-rate decay on plateau, early stopping, multi-seed runs.
- **Ad/context field inference** — Criteo's fields are anonymous, so the split is inferred from conditional entropy between fields, and a probe then picks the candidate ad entity whose ranking depends most on the query.
- **Ranker-distilled two-tower retrieval** — towers trained to imitate DeepFM with a listwise KL loss over uniform and hard-negative slates. A click-trained tower, a popularity list, and a ranker-prior shortlist are included as baselines.
- **Exact FAISS scoring** — each item vector carries its learned bias as an extra coordinate, so inner-product search reproduces the tower score exactly.
- **ONNX export with parity check** — the ranker is exported with a dynamic batch dimension and checked against PyTorch on held-out rows.
- **C++ inference harness** — replays real fine-ranking requests through ONNX Runtime in a naive mode and a tuned mode (explicit intra-op threads, sequential executor, spin-wait, reusable IoBinding buffers), with a thread-count sweep.
- **Live in-browser demo** — pruned copies of the query tower and ranker run in `onnxruntime-web`. Pick an impression, force context fields to missing, and compare the funnel's top-10 with the full-catalog ranking.

## Repository layout

```
adsrank/
  data.py          download a Criteo sample and encode 39 fields
  models.py        LR / FM / DNN / DeepFM rankers, two-tower retriever
  train_ranker.py  training, embedding-L2 sweep, early stopping, seeds
  fields.py        infer candidate ad entities from conditional entropy
  split_probe.py   pick the ad entity whose ranking depends most on the query
  retrieval.py     catalog, oracle, click-trained vs distilled towers, FAISS IVF
  export.py        ONNX export, parity check, latency replay file
  bench_py.py      PyTorch eager and onnxruntime-Python baselines
  export_site.py   pruned browser models and evaluation-board payload
cpp/
  bench.cpp        ONNX Runtime C++ harness (naive vs tuned)
scripts/           end-to-end run, ONNX Runtime setup, C++ build and sweep
docs/              the static site (GitHub Pages)
```

## Setup

Requirements: Linux (tested on Ubuntu under WSL2), Python 3.12, a C++17 compiler, and CMake.

```bash
git clone https://github.com/DevFrancisA/AdsRank && cd AdsRank
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh          # download → train → retrieve → export → benchmark → site assets
python -m http.server -d docs    # open http://localhost:8000
```

`scripts/bench_cpp.sh` downloads the ONNX Runtime C/C++ release into `third_party/` and builds `cpp/bench.cpp` with CMake. Run the benchmark stages on an otherwise idle machine. Set `ADSRANK_DATA` to keep data and artifacts off a slow mount (default `~/adsrank_data`). Benchmark outputs are written to `results/`, which is not tracked.

## Data

Criteo 1TB Click Logs, Hugging Face mirror [`criteo/CriteoClickLogs`](https://huggingface.co/datasets/criteo/CriteoClickLogs), CC BY-NC-SA 4.0.

## License

MIT for the code. The dataset keeps its own license.
