#!/usr/bin/env bash
# Audit every binary under embedded/bin/<platform>/ for dyld load problems
# that would blow up on end-user machines.
#
# Specifically looks for:
#   1. LC_RPATH entries that point into build-host-specific paths
#      (e.g. /Users/runner/... on GitHub Actions, /Users/kouko/... locally).
#      Those paths do not exist on the user's machine and cause
#      "dyld: Library not loaded" at spawn time.
#   2. Dynamic dependencies resolved via @rpath where no shipped companion
#      dylib of that name exists next to the binary. A binary that
#      references @rpath/libfoo.dylib is only OK if libfoo.dylib is
#      actually bundled in the same embedded/bin directory.
#
# System dylibs (/usr/lib/... and /System/Library/...) are allowed. Weak
# links (marked "weak" by otool) are also allowed because dyld is
# tolerant of them being missing.
#
# Run locally after `make deps`, or in CI to fail the release pipeline
# before shipping a broken binary. The original sherpa-onnx bug and the
# whisper-cli bug would both have been caught by this script.
#
# Usage: bash scripts/audit-embedded-binaries.sh [embedded/bin/<platform>]
#        (defaults to embedded/bin/darwin-<arch> for the host)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ $# -ge 1 ]]; then
  TARGET_DIR="$1"
else
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64|aarch64) ARCH="arm64" ;;
    x86_64|amd64)  ARCH="amd64" ;;
  esac
  TARGET_DIR="$PROJECT_ROOT/embedded/bin/darwin-$ARCH"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "audit: target directory not found: $TARGET_DIR" >&2
  echo "audit: run 'make deps' first" >&2
  exit 1
fi

echo "==> Auditing embedded binaries in $TARGET_DIR"

FAIL=0

# Collect the set of shipped dylib filenames so we can check @rpath deps
# against it. Includes any .dylib in the same dir.
shopt -s nullglob
SHIPPED_DYLIBS=()
for f in "$TARGET_DIR"/*.dylib; do
  SHIPPED_DYLIBS+=("$(basename "$f")")
done
shopt -u nullglob

is_shipped() {
  local name="$1"
  local shipped
  for shipped in "${SHIPPED_DYLIBS[@]:-}"; do
    [[ "$shipped" == "$name" ]] && return 0
  done
  return 1
}

# Mach-O magic bytes: 0xcafebabe (fat), 0xfeedface (32-bit), 0xfeedfacf
# (64-bit), and their byte-swapped counterparts. We skip non-Mach-O files
# like model blobs (ggml-silero-v6.2.0.bin).
is_macho() {
  local file="$1"
  local magic
  magic="$(xxd -p -l 4 "$file" 2>/dev/null || true)"
  case "$magic" in
    cafebabe|bebafeca|feedface|cefaedfe|feedfacf|cffaedfe) return 0 ;;
    *) return 1 ;;
  esac
}

check_binary() {
  local bin="$1"
  local name
  name="$(basename "$bin")"
  echo "    $name"

  local bad_rpaths
  bad_rpaths="$(otool -l "$bin" 2>/dev/null \
    | awk '/cmd LC_RPATH/{flag=1; next} flag && /path /{print $2; flag=0}' \
    | grep -E '^/Users/|^/home/|/pkg/mod/|whisper\.cpp/build|sherpa-onnx-go-macos' \
    || true)"
  if [[ -n "$bad_rpaths" ]]; then
    echo "      ERROR: build-host-specific LC_RPATH entries:"
    while IFS= read -r rpath; do
      echo "        $rpath"
    done <<< "$bad_rpaths"
    FAIL=1
  fi

  # otool -L format:
  #   <binary>:
  #     /usr/lib/libfoo.dylib (compatibility version ..., current version ...)
  #     @rpath/libbar.dylib (compatibility version ..., current version ..., weak)
  #
  # Weak deps are skipped because dyld tolerates them being missing.
  # Match on the whole line first to filter out weak entries, then
  # extract the dep name.
  local unresolved
  unresolved="$(otool -L "$bin" 2>/dev/null | tail -n +2 \
    | grep -E '^[[:space:]]*@rpath/' \
    | grep -v ', weak)$' \
    | awk '{print $1}' \
    || true)"

  local dep
  while IFS= read -r dep; do
    [[ -z "$dep" ]] && continue
    local dep_name
    dep_name="${dep#@rpath/}"
    if ! is_shipped "$dep_name"; then
      echo "      ERROR: unresolved @rpath dep: $dep (no $dep_name in $TARGET_DIR)"
      FAIL=1
    fi
  done <<< "$unresolved"
}

for f in "$TARGET_DIR"/*; do
  [[ -f "$f" ]] || continue
  [[ "$(basename "$f")" == .* ]] && continue
  if is_macho "$f"; then
    check_binary "$f"
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo
  echo "audit: FAILED — one or more embedded binaries have bad dyld metadata."
  echo "audit: these binaries will abort with 'Library not loaded' on end-user machines."
  exit 1
fi

echo "==> Audit passed"
