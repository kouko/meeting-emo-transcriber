#!/usr/bin/env bash
set -euo pipefail

ONNX_VERSION="1.19.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/darwin-arm64"
OUTPUT_FILE="$OUTPUT_DIR/libonnxruntime.dylib"
TMP_DIR="$PROJECT_ROOT/.build/onnxruntime"

echo "=== Downloading ONNX Runtime v$ONNX_VERSION ==="

if [ -f "$OUTPUT_FILE" ]; then
    echo "libonnxruntime.dylib already exists at $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

TARBALL="onnxruntime-osx-arm64-$ONNX_VERSION.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/$TARBALL"

if [ ! -f "$TMP_DIR/$TARBALL" ]; then
    echo "Downloading from $URL..."
    curl -L -o "$TMP_DIR/$TARBALL" "$URL"
fi

echo "Extracting..."
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

DYLIB=$(find "$TMP_DIR" -name "libonnxruntime.$ONNX_VERSION.dylib" -type f | head -1)
if [ -z "$DYLIB" ]; then
    DYLIB=$(find "$TMP_DIR" -name "libonnxruntime.dylib" -type f | head -1)
fi

if [ -z "$DYLIB" ]; then
    echo "ERROR: Could not find libonnxruntime.dylib in extracted archive"
    exit 1
fi

cp "$DYLIB" "$OUTPUT_FILE"
chmod 644 "$OUTPUT_FILE"

echo "libonnxruntime.dylib installed: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
