#!/usr/bin/env bash
set -euo pipefail

WHISPER_VERSION="v1.7.3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/darwin-arm64"
BUILD_DIR="$PROJECT_ROOT/.build/whisper.cpp"

echo "=== Building whisper-cli $WHISPER_VERSION ==="

if [ -d "$BUILD_DIR" ]; then
    echo "Using existing whisper.cpp at $BUILD_DIR"
    cd "$BUILD_DIR"
    git fetch --tags
else
    echo "Cloning whisper.cpp..."
    mkdir -p "$(dirname "$BUILD_DIR")"
    git clone https://github.com/ggml-org/whisper.cpp.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

git checkout "$WHISPER_VERSION"

echo "Building with Metal GPU support..."
cmake -B build \
    -DGGML_METAL=1 \
    -DGGML_METAL_EMBED_LIBRARY=1 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"

mkdir -p "$OUTPUT_DIR"
cp build/bin/whisper-cli "$OUTPUT_DIR/whisper-cli"
chmod 755 "$OUTPUT_DIR/whisper-cli"

echo "whisper-cli built: $OUTPUT_DIR/whisper-cli"
ls -lh "$OUTPUT_DIR/whisper-cli"

VAD_MODEL="$OUTPUT_DIR/ggml-silero-v6.2.0.bin"
if [ ! -f "$VAD_MODEL" ]; then
    echo "Downloading Silero VAD model..."
    bash "$BUILD_DIR/models/download-vad-model.sh"
    cp "$BUILD_DIR/models/ggml-silero-v6.2.0.bin" "$VAD_MODEL"
fi
echo "VAD model: $VAD_MODEL"
