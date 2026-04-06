#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/darwin-arm64"
SWIFT_PROJECT="$PROJECT_ROOT/tools/metr-denoise"

echo "=== Building metr-denoise (DeepFilterNet3 speech enhancement) ==="

cd "$SWIFT_PROJECT"
swift build -c release 2>&1

mkdir -p "$OUTPUT_DIR"
cp .build/release/metr-denoise "$OUTPUT_DIR/metr-denoise"
chmod 755 "$OUTPUT_DIR/metr-denoise"

echo "metr-denoise built: $OUTPUT_DIR/metr-denoise"
ls -lh "$OUTPUT_DIR/metr-denoise"
