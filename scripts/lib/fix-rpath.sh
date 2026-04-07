#!/usr/bin/env bash
# Shared helpers for rewriting LC_RPATH in Mach-O binaries that cgo-link
# sherpa-onnx-go-macos. Currently sourced by
# scripts/build-sherpa-sidecar.sh only; kept as a separate library so
# future Mach-O post-processing can reuse the logic without duplicating
# the otool + install_name_tool + codesign dance.
#
# The problem: sherpa-onnx-go-macos bakes `-Wl,-rpath,${SRCDIR}/lib/...`
# into cgo directives, so the linker embeds the build host's Go module
# cache path into the binary's LC_RPATH. That path does not exist on
# user machines. This helper deletes the stale rpath and adds a
# relocatable one chosen by the caller (e.g. @executable_path/.).
#
# Usage (from a script that set -euo pipefail):
#
#   source "$(dirname "$0")/lib/fix-rpath.sh"
#
#   # Find the upstream module's dylib dir without hardcoding the version.
#   SRC_LIB_DIR="$(sherpa_dylib_dir)"
#
#   # Copy dylibs into a destination dir (preserves the versioned symlink).
#   copy_sherpa_dylibs "$SRC_LIB_DIR" "$DEST_LIB_DIR"
#
#   # Delete any rpath pointing into a Go module cache, add the new one,
#   # re-sign (install_name_tool breaks the ad-hoc signature), and assert
#   # no stale rpath remains.
#   rewrite_rpath "$BINARY_PATH" "$NEW_RPATH"

set -euo pipefail

SHERPA_MODULE="github.com/k2-fsa/sherpa-onnx-go-macos"

# sherpa_dylib_dir prints the absolute path to the platform-specific dylib
# directory inside the cached module. It asks the Go toolchain for the
# module location so we never hardcode a version.
sherpa_dylib_dir() {
  local arch dylib_subdir module_dir
  arch="$(uname -m)"
  case "$arch" in
    arm64|aarch64) dylib_subdir="lib/aarch64-apple-darwin" ;;
    x86_64|amd64)  dylib_subdir="lib/x86_64-apple-darwin" ;;
    *)
      echo "fix-rpath: unsupported arch: $arch" >&2
      return 1
      ;;
  esac

  module_dir="$(go list -m -f '{{.Dir}}' "$SHERPA_MODULE" 2>/dev/null || true)"
  if [[ -z "$module_dir" || ! -d "$module_dir" ]]; then
    echo "fix-rpath: could not locate $SHERPA_MODULE via 'go list -m' (try 'go mod download')" >&2
    return 1
  fi

  local src_lib_dir="$module_dir/$dylib_subdir"
  if [[ ! -d "$src_lib_dir" ]]; then
    echo "fix-rpath: dylib directory not found: $src_lib_dir" >&2
    return 1
  fi

  printf '%s\n' "$src_lib_dir"
}

# copy_sherpa_dylibs copies the dylibs metr needs into a destination
# directory.
#
# Upstream sherpa-onnx-go-macos ships two identical copies of the onnxruntime
# dylib (libonnxruntime.dylib and libonnxruntime.X.Y.Z.dylib, 35 MB each
# and bit-for-bit identical — not a symlink). The sherpa C++ API and all
# dependents reference `@rpath/libonnxruntime.X.Y.Z.dylib`, so the
# unversioned file is dead weight at runtime. We copy only the versioned
# one and save ~35 MB of embedded payload.
#
# The versioned onnxruntime filename is discovered dynamically so future
# sherpa-onnx upgrades don't require a script change.
copy_sherpa_dylibs() {
  local src_dir="$1"
  local dest_dir="$2"

  mkdir -p "$dest_dir"
  rm -f "$dest_dir"/*.dylib

  # Discover the versioned libonnxruntime dynamically.
  local versioned_ort
  versioned_ort="$(cd "$src_dir" && ls libonnxruntime.*.dylib 2>/dev/null | grep -v '^libonnxruntime\.dylib$' | head -n1 || true)"
  if [[ -z "$versioned_ort" ]]; then
    echo "fix-rpath: no versioned libonnxruntime.*.dylib found in $src_dir" >&2
    return 1
  fi

  local required=("libsherpa-onnx-c-api.dylib" "$versioned_ort")

  local dylib
  for dylib in "${required[@]}"; do
    local src="$src_dir/$dylib"
    if [[ ! -f "$src" ]]; then
      echo "fix-rpath: missing required dylib: $src" >&2
      return 1
    fi
    cp "$src" "$dest_dir/"
    echo "    copied $dylib"
  done
}

# rewrite_rpath deletes any LC_RPATH entries that point into a Go module
# cache, adds a new relocatable rpath, and re-signs the binary with an
# ad-hoc signature (install_name_tool invalidates the original). It
# verifies no stale rpath remains; failure aborts the script.
#
# Arguments:
#   $1  path to the Mach-O binary to rewrite (mutated in place)
#   $2  new rpath value (e.g. "@executable_path/../lib" or "@executable_path/.")
rewrite_rpath() {
  local binary="$1"
  local new_rpath="$2"

  if [[ ! -f "$binary" ]]; then
    echo "fix-rpath: binary not found: $binary" >&2
    return 1
  fi

  chmod u+w "$binary"

  echo "==> Rewriting LC_RPATH on $binary"

  # Enumerate every LC_RPATH entry currently baked into the binary. otool
  # output for LC_RPATH:
  #   cmd LC_RPATH
  #   cmdsize 64
  #   path /Users/runner/go/pkg/mod/... (offset 12)
  local old_rpaths
  old_rpaths="$(otool -l "$binary" \
    | awk '/cmd LC_RPATH/{flag=1; next} flag && /path /{print $2; flag=0}')"

  if [[ -z "$old_rpaths" ]]; then
    echo "warning: no LC_RPATH entries found (unexpected for a cgo binary)" >&2
  fi

  local removed_any=0
  local rpath
  while IFS= read -r rpath; do
    [[ -z "$rpath" ]] && continue
    # Only touch rpaths that point into a Go module cache. Preserve any
    # unrelated entries (there shouldn't be any, but be defensive).
    if [[ "$rpath" == *"/pkg/mod/"* || "$rpath" == *"sherpa-onnx-go-macos"* ]]; then
      echo "    removing rpath: $rpath"
      install_name_tool -delete_rpath "$rpath" "$binary"
      removed_any=1
    else
      echo "    keeping rpath:  $rpath"
    fi
  done <<< "$old_rpaths"

  if [[ "$removed_any" -eq 0 ]]; then
    echo "warning: no module-cache rpath was removed; binary may already be clean" >&2
  fi

  echo "    adding rpath:   $new_rpath"
  install_name_tool -add_rpath "$new_rpath" "$binary"

  # Re-sign with the ad-hoc identity. install_name_tool invalidates the
  # original signature, and on Apple Silicon an unsigned Mach-O is killed
  # by the kernel before dyld even reports the load error.
  if command -v codesign >/dev/null 2>&1; then
    echo "==> Re-signing binary (ad-hoc)"
    codesign --force --sign - "$binary"
  fi

  # Verify: no /Users/runner or /pkg/mod/, and the new rpath is present.
  echo "==> Verifying"
  local post_rpaths
  post_rpaths="$(otool -l "$binary" \
    | awk '/cmd LC_RPATH/{flag=1; next} flag && /path /{print $2; flag=0}')"
  echo "    final rpaths:"
  echo "$post_rpaths" | sed 's/^/      /'

  if echo "$post_rpaths" | grep -qE '/Users/runner|/pkg/mod/'; then
    echo "fix-rpath: stale runner/module-cache rpath still present" >&2
    return 1
  fi
  if ! echo "$post_rpaths" | grep -qx "$new_rpath"; then
    echo "fix-rpath: expected rpath not added: $new_rpath" >&2
    return 1
  fi
}
