# AdsRank

**A three-stage click-through-rate ranking stack on Criteo — two-tower retrieval, FAISS candidate recall, DeepFM fine-ranking, ONNX Runtime C++ serving.**

**Live datasheet → https://devfrancisa.github.io/AdsRank/** (the funnel runs in your browser on the evaluation board, §8)

## Results

5,000,000 Criteo impressions (4M / 0.5M / 0.5M), one AMD Ryzen 7 9800X3D, Ubuntu 26.04 on WSL2.

**Ranker — feature-interaction ablation** (test split, best-validation checkpoint, embedding L2 chosen on validation)

| Model | Linear | FM | Deep MLP | Test AUC | Test log loss | vs LR |
|---|:-:|:-:|:-:|--:|--:|--:|
| Logistic regression | ● | | | 0.7602 | 0.12686 | — |
| FM | ● | ● | | 0.7682 | 0.12574 | −0.88% |
| DNN | ● | | ● | 0.7719 | 0.12533 | −1.21% |
| **DeepFM** | ● | ● | ● | **0.7719** | **0.12523** | **−1.28%** |

Across 3 seeds DeepFM spans 0.7702–0.7719 AUC and LR 0.7595–0.7602. On this sample the deep tower carries almost all of the gain; the FM term mostly gets there an epoch sooner.

**Retrieval funnel** — retention@10 against DeepFM scoring all 9,919 catalog items, 1,000 held-out queries

| Ranker calls per request | Distilled two-tower + FAISS | One global shortlist | Click-trained two-tower | Popularity |
|---|--:|--:|--:|--:|
| 10 (0.1%) | 91.9% | 56.0% | 0.4% | 0.0% |
| 20 (0.2%) | 99.8% | 69.7% | 0.8% | 0.0% |
| 99 (1%) | 100% | 90.4% | 3.5% | 0.1% |
| **992 (10%)** | **100%** | 99.3% | 17.6% | 11.0% |

The "global shortlist" baseline (top items by mean DeepFM logit, same list for every query) is the honest reference: on Criteo, DeepFM's item ordering is dominated by item main effects, so the funnel's value is the query-dependent remainder at tight budgets. A click-trained tower learns which ads were *served* where, not what the ranker prefers. FAISS IVF (nlist 100, nprobe 31) recovers 99.75% of the exact shortlist at 0.04 ms/query.

**Serving** — one fine-ranking request = 1 context × 992 candidates, 2,000 replayed requests per configuration

| Path | p50 | p95 | p99 |
|---|--:|--:|--:|
| PyTorch eager | 2.51 ms | 2.73 ms | 2.92 ms |
| ONNX Runtime, Python | 1.14 ms | 1.52 ms | 2.39 ms |
| ONNX Runtime C++, naive (default threads) | 1.15 ms | 1.57 ms | 2.43 ms |
| **ONNX Runtime C++, tuned (12 threads, IoBinding)** | **1.26 ms** | **1.35 ms** | **1.50 ms** |

ONNX vs PyTorch max |Δp| = 1.1e-7 on 20,000 held-out rows. Most of the win is ONNX Runtime itself plus the thread count. At matched thread counts, the tuned session (reused IoBinding buffers, spin-wait, sequential executor) mainly trims the tail — p99 1.69 → 1.50 ms at 12 threads, p95 4.46 → 2.90 ms at 2 threads — while spinning on all 16 logical threads hurts it (p99 4.29 ms).

Every number above is written by the code in this repository into `results/` and rendered from there; nothing is hand-entered.

## What's inside

```
adsrank/
  data.py          download a 5M-impression Criteo sample, encode 39 fields (train-only vocab)
  models.py        LR / FM / DNN / DeepFM rankers, two-tower retriever
  train_ranker.py  training with validation-selected embedding L2, LR decay, early stopping
  fields.py        infer the ad/context field split from conditional entropy
  retrieval.py     catalog, oracle, click-trained vs ranker-distilled towers, FAISS IVF, retention@10
  export.py        ONNX export + parity check + latency replay file
  bench_py.py      PyTorch eager and onnxruntime-Python latency baselines
  export_site.py   report.json, pruned browser models, evaluation-board payload
cpp/
  bench.cpp        ONNX Runtime C++ harness: naive vs tuned (threads, IoBinding, buffer reuse)
docs/              the datasheet site (static; GitHub Pages)
results/           every metric the README and site print
```

## How it works

**Ranker.** Count features are bucketed by ⌊ln²x⌋ (the Kaggle-winning transform), categoricals get a train-only vocabulary with a frequency floor of 10, and all 39 fields index one shared embedding table. DeepFM sums a linear term, the FM pairwise term and a 400-400-400 MLP over the *same* 16-d embeddings. The ablation trains LR, FM, DNN and DeepFM under one protocol: Adam, four validation passes per epoch, LR decay on plateau, early stopping, and embedding L2 picked per configuration on validation log loss. Test metrics are computed once, on the best-validation checkpoint.

**Ad/context split.** Criteo's fields are anonymous, so `fields.py` computes U(J|K) = H(J|K)/H(J) for every pair of categorical fields. The high-cardinality field that (almost) determines the most others becomes the ad key; the fields it determines are ad attributes; everything else is context.

**Retrieval.** The catalog is the 10,000 most frequent ad keys. For held-out queries, an *oracle* runs DeepFM on every catalog item. The funnel instead encodes the query with a two-tower model, retrieves the top-K from a FAISS IVF index, and runs DeepFM only on those K. Retention@10 is the overlap of the two top-10 lists. A click-trained tower (in-batch softmax, logQ correction) is compared with a tower **distilled from DeepFM** with a listwise KL loss over uniform + hard-negative slates. Item vectors carry a learned bias as an extra coordinate (queries append a 1), so FAISS inner product equals the tower score exactly.

**Serving.** The ranker is exported to ONNX (opset 18, dynamic batch, sigmoid in-graph) and checked against PyTorch. Latency is measured on real fine-ranking requests — one context × K candidates — replayed through PyTorch eager, onnxruntime-Python, and a C++ harness in a *naive* mode (default threads, allocation per call) and a *tuned* mode (explicit intra-op threads, sequential executor, spin-wait, one pre-bound input/output buffer pair via `IoBinding`), swept over thread counts.

## Reproduce

Linux (tested on Ubuntu 26.04 under WSL2), Python 3.12, a C++17 compiler and CMake:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh          # ~1.5 h on 16 CPU threads; downloads ~340 MB
python -m http.server -d docs    # view the datasheet at http://localhost:8000
```

`scripts/bench_cpp.sh` fetches the ONNX Runtime C/C++ release into `third_party/` and builds `cpp/bench.cpp` with CMake. Run the latency stages on an otherwise idle machine.

## Data

Criteo 1TB Click Logs, Hugging Face mirror [`criteo/CriteoClickLogs`](https://huggingface.co/datasets/criteo/CriteoClickLogs), CC BY-NC-SA 4.0. The first 5,000,000 rows of the day-2015-02-15 shards are split 80/10/10 at random with a fixed seed. Criteo subsamples negatives in this release, so the base CTR (~3.2%) and absolute AUC are not comparable to the Kaggle challenge split.

## License

MIT for the code. The dataset keeps its own license.
