#!/usr/bin/env bash
# Build the C++ harness and sweep thread counts for both modes.
# Writes results/latency_cpp.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${ADSRANK_DATA:-$HOME/adsrank_data}"
ITERS="${ITERS:-2000}"
BENCH="$(bash scripts/build_cpp.sh)"

out=results/latency_cpp.jsonl
: > "$out"
# default-thread naive run is the "just call ORT from C++" reference point
"$BENCH" "$DATA/artifacts/deepfm.onnx" "$DATA/artifacts/requests.bin" naive 0 "$ITERS" | tee -a "$out"
for t in 1 2 4 6 8 12 16; do
  for mode in naive tuned; do
    "$BENCH" "$DATA/artifacts/deepfm.onnx" "$DATA/artifacts/requests.bin" "$mode" "$t" "$ITERS" | tee -a "$out"
  done
done
