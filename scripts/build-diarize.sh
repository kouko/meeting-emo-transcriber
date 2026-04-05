#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/darwin-arm64"
SWIFT_PROJECT="$PROJECT_ROOT/tools/metr-diarize"

echo "=== Building metr-diarize (FluidAudio speaker diarization) ==="

cd "$SWIFT_PROJECT"
swift build -c release 2>&1

mkdir -p "$OUTPUT_DIR"
cp .build/release/metr-diarize "$OUTPUT_DIR/metr-diarize"
chmod 755 "$OUTPUT_DIR/metr-diarize"

echo "metr-diarize built: $OUTPUT_DIR/metr-diarize"
ls -lh "$OUTPUT_DIR/metr-diarize"
