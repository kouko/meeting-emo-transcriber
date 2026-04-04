package asr

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type WhisperConfig struct {
	BinPath      string
	ModelPath    string
	VADModelPath string
	Language     string // "auto" | "zh-TW" | "zh" | "en" | "ja"
	Threads      int
}

// whisperLang converts our language codes to whisper-cli compatible codes.
// whisper-cli uses lowercase ISO codes: "zh", "en", "ja", "auto".
var whisperLang = map[string]string{
	"zh-TW": "zh",
	"zh":    "zh",
	"en":    "en",
	"ja":    "ja",
	"auto":  "auto",
}

// buildWhisperArgs constructs the whisper-cli argument list.
func buildWhisperArgs(cfg WhisperConfig, wavPath, outputBase string) []string {
	lang := cfg.Language
	if mapped, ok := whisperLang[lang]; ok {
		lang = mapped
	}
	args := []string{
		"-m", cfg.ModelPath,
		"-f", wavPath,
		"-l", lang,
		"-t", strconv.Itoa(cfg.Threads),
		"-osrt",
		"-of", outputBase,
	}
	// VAD support: only add flags if whisper-cli version supports them.
	// v1.7.3 does not have --vad; later versions do.
	// For now, VAD flags are disabled to maintain v1.7.3 compatibility.
	// TODO: re-enable when upgrading whisper.cpp to a version with Silero VAD support.
	// if cfg.VADModelPath != "" {
	// 	args = append(args, "--vad", "-vm", cfg.VADModelPath)
	// }
	return args
}

// Transcribe runs whisper-cli and returns parsed ASR results.
func Transcribe(cfg WhisperConfig, wavPath string) ([]types.ASRResult, error) {
	tmpDir, err := os.MkdirTemp("", "met-asr-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputBase := filepath.Join(tmpDir, "output")
	args := buildWhisperArgs(cfg, wavPath, outputBase)

	cmd := exec.Command(cfg.BinPath, args...)
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("whisper-cli failed: %w", err)
	}

	srtPath := outputBase + ".srt"
	srtData, err := os.ReadFile(srtPath)
	if err != nil {
		return nil, fmt.Errorf("read SRT output: %w", err)
	}

	return ParseSRT(string(srtData), cfg.Language)
}
