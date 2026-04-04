package models

import "strings"

// ModelInfo describes a downloadable model asset.
type ModelInfo struct {
	Name      string
	URL       string
	SHA256    string
	Size      int64  // bytes, for progress display
	Category  string // "asr" | "vad" | "speaker" | "emotion"
	IsArchive bool   // true if the download is a compressed archive (e.g. .tar.bz2)
}

// Registry is the catalogue of all known models.
var Registry = map[string]ModelInfo{
	"ggml-large-v3": {
		Name:     "ggml-large-v3",
		URL:      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
		SHA256:   "", // Populated after first download
		Size:     3100000000,
		Category: "asr",
	},
	"silero-vad-v6.2.0": {
		Name:     "silero-vad-v6.2.0",
		URL:      "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
		SHA256:   "",
		Size:     900000,
		Category: "vad",
	},
	"ggml-breeze-asr-25-q5k": {
		Name:     "ggml-breeze-asr-25-q5k",
		URL:      "https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp/resolve/main/ggml-model-q5_k.bin",
		SHA256:   "",
		Size:     1100000000,
		Category: "asr",
	},
	"ggml-belle-zh": {
		Name:     "ggml-belle-zh",
		URL:      "https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh-ggml/resolve/main/ggml-model.bin",
		SHA256:   "",
		Size:     1500000000,
		Category: "asr",
	},
	"ggml-kotoba-whisper-v2.0": {
		Name:     "ggml-kotoba-whisper-v2.0",
		URL:      "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/main/ggml-kotoba-whisper-v2.0.bin",
		SHA256:   "",
		Size:     1520000000,
		Category: "asr",
	},
	"campplus-sv-zh-cn": {
		Name:     "campplus-sv-zh-cn",
		URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx",
		SHA256:   "",
		Size:     30000000,
		Category: "speaker",
	},
	"sensevoice-small-int8": {
		Name:      "sensevoice-small-int8",
		URL:       "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
		SHA256:    "",
		Size:      228000000,
		Category:  "emotion",
		IsArchive: true,
	},
}

// ResolveASRModel returns the model name for a given language.
func ResolveASRModel(language string) string {
	switch language {
	case "zh-TW":
		return "ggml-breeze-asr-25-q5k"
	case "zh":
		return "ggml-belle-zh"
	case "ja":
		return "ggml-kotoba-whisper-v2.0"
	default: // auto, en
		return "ggml-large-v3"
	}
}

// modelFilename returns the local filename (or directory name for archives) for a model.
func modelFilename(name, url string) string {
	switch {
	case strings.HasSuffix(url, ".tar.bz2"):
		return name // directory name for extracted archive
	case strings.HasSuffix(url, ".onnx"):
		return name + ".onnx"
	default:
		return name + ".bin"
	}
}
