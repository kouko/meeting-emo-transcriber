package asr

import (
	"bytes"
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

// buildWhisperArgs constructs the whisper-cli argument list.
func buildWhisperArgs(cfg WhisperConfig, wavPath, outputBase string) []string {
	args := []string{
		"-m", cfg.ModelPath,
		"-f", wavPath,
		"-l", cfg.Language,
		"-t", strconv.Itoa(cfg.Threads),
		"-osrt",
		"-of", outputBase,
		"--no-prints",
	}
	if cfg.VADModelPath != "" {
		args = append(args, "--vad", "-vm", cfg.VADModelPath)
	}
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
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("whisper-cli failed: %w\nstderr: %s", err, stderr.String())
	}

	srtPath := outputBase + ".srt"
	srtData, err := os.ReadFile(srtPath)
	if err != nil {
		return nil, fmt.Errorf("read SRT output: %w", err)
	}

	return ParseSRT(string(srtData), cfg.Language)
}
