package models

// ModelInfo describes a downloadable model asset.
type ModelInfo struct {
	Name     string
	URL      string
	SHA256   string
	Size     int64  // bytes, for progress display
	Category string // "asr" | "vad" | "speaker" | "emotion"
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
		URL:      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-v6.2.0.bin",
		SHA256:   "",
		Size:     2000000,
		Category: "vad",
	},
}

// ResolveASRModel returns the model name to use for a given language.
// Phase 2: all languages use ggml-large-v3.
func ResolveASRModel(language string) string {
	return "ggml-large-v3"
}
