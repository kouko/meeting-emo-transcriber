#!/usr/bin/env bash
# Update Homebrew formula with new version and SHA256 checksum.
# Usage: update-formula.sh <version> <checksums-file> <formula-file>
#
# Example:
#   update-formula.sh 1.0.0 checksums.txt homebrew-tap/Formula/metr.rb

set -euo pipefail

VERSION="${1:?Usage: update-formula.sh <version> <checksums-file> <formula-file>}"
CHECKSUMS_FILE="${2:?Missing checksums file}"
FORMULA_FILE="${3:?Missing formula file}"

if [[ ! -f "$CHECKSUMS_FILE" ]]; then
  echo "Error: checksums file not found: $CHECKSUMS_FILE" >&2
  exit 1
fi

SHA_DARWIN_ARM64=$(grep "darwin-arm64" "$CHECKSUMS_FILE" | awk '{print $1}')

if [[ -z "$SHA_DARWIN_ARM64" ]]; then
  echo "Error: could not find SHA256 for darwin-arm64 in $CHECKSUMS_FILE" >&2
  exit 1
fi

cat > "$FORMULA_FILE" <<RUBY
class Metr < Formula
  desc "Meeting transcriber with speaker identification and emotion recognition"
  homepage "https://github.com/kouko/meeting-emo-transcriber"
  version "${VERSION}"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/kouko/meeting-emo-transcriber/releases/download/v#{version}/metr-darwin-arm64.tar.gz"
      sha256 "${SHA_DARWIN_ARM64}"
    end
  end

  def install
    bin.install "metr"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/metr --version 2>&1")
  end
end
RUBY

echo "Updated $FORMULA_FILE to v${VERSION}"
echo "  darwin-arm64: ${SHA_DARWIN_ARM64}"
