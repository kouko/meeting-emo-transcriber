#!/bin/bash
# build-ffmpeg.sh — Build minimal LGPL ffmpeg from source
#
# Builds ffmpeg with only the features needed for metr:
# - Audio format conversion (WAV, MP3, M4A, FLAC, OGG, OPUS, AAC)
# - Audio resampling (16kHz mono for whisper)
# - Audio volume detection (volumedetect filter)
# - Audio normalization (loudnorm filter)
#
# No --enable-gpl: output is LGPL-licensed, safe to embed in MIT project.
#
# Prerequisites:
#   xcode-select --install && brew install cmake nasm pkg-config
#
# Usage:
#   ./scripts/build-ffmpeg.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TARGET_OS="darwin"
TARGET_ARCH="arm64"

OUTPUT_DIR="$PROJECT_DIR/embedded/bin/${TARGET_OS}-${TARGET_ARCH}"
OUTPUT_PATH="${OUTPUT_DIR}/ffmpeg"
mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_PATH" ]; then
    echo "[INFO] ffmpeg already exists at $OUTPUT_PATH, skipping build"
    echo "[INFO] Delete the file to rebuild"
    exit 0
fi

# Temp build directory
TMPBASE="${TMPDIR:-/tmp}"
BUILD_DIR="${TMPBASE}/metr-ffmpeg-build-$$"
mkdir -p "$BUILD_DIR"

cleanup() {
    echo "[INFO] Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

echo "[INFO] Building minimal LGPL ffmpeg for ${TARGET_OS}-${TARGET_ARCH}..."

# Clone ffmpeg
FFMPEG_VERSION="n7.1"
echo "[INFO] Cloning ffmpeg ${FFMPEG_VERSION}..."
git clone --depth 1 --branch "$FFMPEG_VERSION" https://github.com/FFmpeg/FFmpeg.git "$BUILD_DIR/ffmpeg"
cd "$BUILD_DIR/ffmpeg"

# Configure — minimal build for audio conversion only (LGPL, no --enable-gpl)
CONFIGURE_FLAGS=(
    --prefix="$BUILD_DIR/install"
    --enable-static
    --disable-shared
    --disable-doc
    --disable-htmlpages
    --disable-manpages
    --disable-podpages
    --disable-txtpages
    # Only build ffmpeg binary
    --disable-programs
    --enable-ffmpeg
    --disable-ffplay
    --disable-ffprobe
    # Disable video (we only need audio)
    --disable-avdevice
    --disable-swscale
    --disable-postproc
    # Disable unnecessary features
    --disable-network
    --disable-debug
    --disable-runtime-cpudetect
    # Disable autodetection of external libraries that might exist on the
    # build host but not on end-user machines. Without these flags ffmpeg's
    # ./configure happily picks up Homebrew-installed libs (e.g. libX11)
    # and bakes their absolute paths into the resulting binary, which then
    # fails with "Library not loaded: /opt/homebrew/opt/libx11/lib/..."
    # on user machines that don't have libx11 installed via Homebrew.
    # We don't use any of these features (audio-only, no video display).
    --disable-xlib
    --disable-libxcb
    --disable-libxcb-shm
    --disable-libxcb-shape
    --disable-libxcb-xfixes
    --disable-sdl2
    # Enable what we need
    --enable-small
    # macOS: use native AudioToolbox for audio decoding
    --enable-audiotoolbox
    --disable-videotoolbox
    --arch=aarch64
    --enable-cross-compile
    --target-os=darwin
)

echo "[INFO] Configuring ffmpeg..."
./configure "${CONFIGURE_FLAGS[@]}"

echo "[INFO] Building ffmpeg..."
make -j"$(sysctl -n hw.ncpu)"

echo "[INFO] Installing ffmpeg binary..."
cp ffmpeg "$OUTPUT_PATH"
chmod +x "$OUTPUT_PATH"

echo "[SUCCESS] ffmpeg built at $OUTPUT_PATH"
ls -lh "$OUTPUT_PATH"
"$OUTPUT_PATH" -version 2>/dev/null | head -1 || echo "[WARN] Could not verify version"
