#!/usr/bin/env bash
# Fetch the ONNX Runtime C/C++ release the benchmark links against.
set -euo pipefail
cd "$(dirname "$0")/.."
VER="${ORT_VERSION:-1.31.0}"
DIR="third_party/onnxruntime-linux-x64-$VER"
if [ ! -d "$DIR" ]; then
  mkdir -p third_party
  curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v$VER/onnxruntime-linux-x64-$VER.tgz" \
    | tar xz -C third_party
fi
echo "$PWD/$DIR"
