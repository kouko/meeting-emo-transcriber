#!/usr/bin/env bash
# Build the metr-sherpa sidecar binary plus the sherpa-onnx dylibs it
# needs, and place them into embedded/bin/<platform>/ so that go:embed
# can pick them up when building the main metr binary.
#
# The main metr binary embeds the sidecar + dylibs and extracts them to
# ~/.metr/bin/ on first run (same mechanism as whisper-cli, ffmpeg,
# metr-diarize, metr-denoise). At runtime the sidecar and its dylibs sit
# in the same flat directory, so the sidecar's LC_RPATH is
# `@executable_path/.` — not `../lib` like the old bundle-dylibs.sh
# layout, because there is no bin/ + lib/ split here.
#
# This script is called from `make build-sherpa-sidecar`, which runs as
# part of `make deps` before the main `go build`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# shellcheck source=lib/fix-rpath.sh
source "$SCRIPT_DIR/lib/fix-rpath.sh"

# Platform detection: the sidecar is a CGO binary so we build for the host
# only. When cross-compilation support is added this will need to grow.
GOOS="$(go env GOOS)"
GOARCH="$(go env GOARCH)"
PLATFORM="${GOOS}-${GOARCH}"
OUTPUT_DIR="$PROJECT_ROOT/embedded/bin/$PLATFORM"

echo "=== Building metr-sherpa sidecar for $PLATFORM ==="

mkdir -p "$OUTPUT_DIR"

# Build the sidecar into a temp location first, then move the verified
# artifact into embedded/bin. This keeps go:embed from seeing a
# half-rewritten binary if rewrite_rpath fails partway through.
TMP_BIN="$(mktemp -t metr-sherpa.XXXXXX)"
trap 'rm -f "$TMP_BIN"' EXIT

echo "==> Compiling cmd/metr-sherpa"
(
  cd "$PROJECT_ROOT"
  CGO_ENABLED=1 go build -o "$TMP_BIN" ./cmd/metr-sherpa
)

# Locate the upstream dylib directory and copy the dylibs next to where
# the sidecar binary will live. Flat layout: binary and dylibs as peers.
echo "==> Locating sherpa-onnx dylib directory"
SRC_LIB_DIR="$(sherpa_dylib_dir)"
echo "    $SRC_LIB_DIR"

echo "==> Copying dylibs to $OUTPUT_DIR"
copy_sherpa_dylibs "$SRC_LIB_DIR" "$OUTPUT_DIR"

# Rewrite the sidecar's rpath to resolve dylibs in the same directory as
# the binary itself. `@executable_path` is the directory containing the
# running Mach-O, so `@executable_path/.` means "look for dylibs right
# here". This is the correct rpath for the go:embed extract layout
# where all embedded files end up in a single flat dir (~/.metr/bin/).
rewrite_rpath "$TMP_BIN" "@executable_path/."

# Move into place only after successful rewrite + verification.
mv "$TMP_BIN" "$OUTPUT_DIR/metr-sherpa"
chmod 755 "$OUTPUT_DIR/metr-sherpa"
trap - EXIT

# Smoke test: spawn the sidecar, close its stdin, verify clean exit. This
# proves dyld can resolve the dylibs via the new rpath without needing
# any model files.
echo "==> Smoke test: spawn sidecar and close stdin"
if "$OUTPUT_DIR/metr-sherpa" < /dev/null; then
  echo "    sidecar exited cleanly on EOF"
else
  echo "error: sidecar failed smoke test" >&2
  exit 1
fi

echo "==> Sidecar build complete"
echo "    $OUTPUT_DIR/metr-sherpa"
ls -lh "$OUTPUT_DIR" | awk '/\.dylib$|metr-sherpa/ {print "    " $0}'
