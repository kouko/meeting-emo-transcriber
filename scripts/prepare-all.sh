#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/darwin-arm64"

echo "=========================================="
echo "  Preparing all embedded binaries"
echo "=========================================="
echo ""

bash "$SCRIPT_DIR/build-whisper.sh"
echo ""
bash "$SCRIPT_DIR/download-ffmpeg.sh"
echo ""

echo "=========================================="
echo "  Verification"
echo "=========================================="

PASS=true
for f in whisper-cli ffmpeg; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        SIZE=$(ls -lh "$OUTPUT_DIR/$f" | awk '{print $5}')
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ $f MISSING"
        PASS=false
    fi
done

if [ -f "$OUTPUT_DIR/ggml-silero-v6.2.0.bin" ]; then
    SIZE=$(ls -lh "$OUTPUT_DIR/ggml-silero-v6.2.0.bin" | awk '{print $5}')
    echo "  ✓ ggml-silero-v6.2.0.bin ($SIZE)"
else
    echo "  ✗ ggml-silero-v6.2.0.bin MISSING"
    PASS=false
fi

echo ""
if [ "$PASS" = true ]; then
    echo "All binaries ready. You can now build with:"
    echo "  go build -tags embed -o meeting-emo-transcriber ./cmd/main.go"
else
    echo "ERROR: Some binaries are missing. Fix the errors above and re-run."
    exit 1
fi
