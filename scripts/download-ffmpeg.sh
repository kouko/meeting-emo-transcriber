#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/binaries/darwin-arm64"
OUTPUT_FILE="$OUTPUT_DIR/ffmpeg"

echo "=== Downloading ffmpeg ==="

if [ -f "$OUTPUT_FILE" ]; then
    echo "ffmpeg already exists at $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

BREW_FFMPEG="$(brew --prefix ffmpeg 2>/dev/null || true)/bin/ffmpeg"
if [ -f "$BREW_FFMPEG" ]; then
    echo "Copying ffmpeg from Homebrew: $BREW_FFMPEG"
    cp "$BREW_FFMPEG" "$OUTPUT_FILE"
    chmod 755 "$OUTPUT_FILE"
    echo "ffmpeg copied: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

echo "Homebrew ffmpeg not found. Please install ffmpeg:"
echo "  brew install ffmpeg"
exit 1
