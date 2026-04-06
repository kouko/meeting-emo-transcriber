class Metr < Formula
  desc "Meeting transcriber with speaker identification and emotion recognition"
  homepage "https://github.com/kouko/meeting-emo-transcriber"
  version "0.0.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/kouko/meeting-emo-transcriber/releases/download/v#{version}/metr-darwin-arm64.tar.gz"
      sha256 "PLACEHOLDER_DARWIN_ARM64"
    end
  end

  def install
    bin.install "metr"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/metr --version 2>&1")
  end
end
