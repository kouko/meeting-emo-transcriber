//go:build embed

package embedded

import _ "embed"

//go:embed ../../embedded/binaries/darwin-arm64/whisper-cli
var embeddedWhisperCLI []byte

//go:embed ../../embedded/binaries/darwin-arm64/ffmpeg
var embeddedFFmpeg []byte

//go:embed ../../embedded/binaries/darwin-arm64/libonnxruntime.dylib
var embeddedONNXRuntime []byte

func init() {
	whisperCLIData = func() []byte { return embeddedWhisperCLI }
	ffmpegData = func() []byte { return embeddedFFmpeg }
	onnxRuntimeData = func() []byte { return embeddedONNXRuntime }
}
