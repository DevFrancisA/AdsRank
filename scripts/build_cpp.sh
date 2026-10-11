#!/usr/bin/env bash
# Configure + build cpp/bench against the ONNX Runtime release; prints the binary path.
set -euo pipefail
cd "$(dirname "$0")/.."
ORT_ROOT="${ORT_ROOT:-$(bash scripts/setup_ort.sh)}"
BUILD="${BUILD_DIR:-${ADSRANK_DATA:-$HOME/adsrank_data}/build}"
cmake -S cpp -B "$BUILD" -DORT_ROOT="$ORT_ROOT" >/dev/null
cmake --build "$BUILD" -j >&2
echo "$BUILD/bench"
