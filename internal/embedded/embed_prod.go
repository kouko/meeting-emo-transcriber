//go:build embed

package embedded

import _ "embed"

//go:embed ../../embedded/binaries/darwin-arm64/whisper-cli
var embeddedWhisperCLI []byte

//go:embed ../../embedded/binaries/darwin-arm64/ffmpeg
var embeddedFFmpeg []byte

func init() {
	whisperCLIData = func() []byte { return embeddedWhisperCLI }
	ffmpegData = func() []byte { return embeddedFFmpeg }
}
