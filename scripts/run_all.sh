#!/usr/bin/env bash
# End-to-end: data -> rankers -> field split -> retrieval -> ONNX -> latency -> site assets.
# ~1-1.5 h on a 16-thread CPU. Set ADSRANK_DATA to keep data off a slow mount.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

$PY -m adsrank.data --rows 5000000
# embedding-L2 chosen per configuration on validation; extra seeds give MIN/MAX
$PY -m adsrank.train_ranker --arch deepfm lr fm dnn --l2 0 1e-5 3e-5 1e-4
$PY -m adsrank.train_ranker --arch deepfm lr --seeds 2 3   # reuses each arch's selected L2
$PY -m adsrank.fields
$PY -m adsrank.split_probe     # needs the trained ranker
$PY -m adsrank.retrieval
$PY -m adsrank.export
# latency: run on an otherwise idle machine
$PY -m adsrank.bench_py
bash scripts/bench_cpp.sh
$PY -m adsrank.export_site
