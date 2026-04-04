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

# v1.7.3 builds as "main", newer versions as "whisper-cli"
if [ -f build/bin/whisper-cli ]; then
    cp build/bin/whisper-cli "$OUTPUT_DIR/whisper-cli"
elif [ -f build/bin/main ]; then
    cp build/bin/main "$OUTPUT_DIR/whisper-cli"
else
    echo "ERROR: Could not find whisper-cli or main in build/bin/"
    ls build/bin/
    exit 1
fi
chmod 755 "$OUTPUT_DIR/whisper-cli"

echo "whisper-cli built: $OUTPUT_DIR/whisper-cli"
ls -lh "$OUTPUT_DIR/whisper-cli"

VAD_MODEL="$OUTPUT_DIR/ggml-silero-v6.2.0.bin"
if [ ! -f "$VAD_MODEL" ]; then
    echo "Downloading Silero VAD model..."
    VAD_URL="https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"
    curl -L -o "$VAD_MODEL" "$VAD_URL"
fi
echo "VAD model: $VAD_MODEL"
ls -lh "$VAD_MODEL"
