package asr

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type WhisperConfig struct {
	BinPath  string
	ModelPath string
	Language  string // "auto" | "zh-TW" | "zh" | "en" | "ja"
	Threads   int
	Prompt    string // initial prompt for vocabulary/context hints
}

// whisperLang converts our language codes to whisper-cli compatible codes.
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
	if cfg.Prompt != "" {
		args = append(args, "--prompt", cfg.Prompt)
	}
	return args
}

// cacheKey generates a short hash from file path + size + mtime + extra suffix.
func cacheKey(filePath, extra string) (string, error) {
	info, err := os.Stat(filePath)
	if err != nil {
		return "", err
	}
	key := fmt.Sprintf("%s|%d|%d|%s", filePath, info.Size(), info.ModTime().UnixNano(), extra)
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:16]), nil
}

// TranscribeWithCache checks for a cached SRT result before running whisper-cli.
// On cache hit, returns parsed results immediately (~0s instead of ~7min).
// On cache miss, runs whisper-cli and saves the SRT to cache.
// originalPath is used for cache key (stable path), wavPath is the actual WAV to process.
func TranscribeWithCache(cfg WhisperConfig, wavPath, originalPath string) ([]types.ASRResult, error) {
	cacheDir := filepath.Join(embedded.CacheDir(), "cache")
	os.MkdirAll(cacheDir, 0755)

	key, err := cacheKey(originalPath, cfg.Language)
	if err != nil {
		// Can't compute cache key — fall through to normal transcription
		return Transcribe(cfg, wavPath)
	}

	srtCachePath := filepath.Join(cacheDir, key+".srt")

	// Cache hit
	if data, err := os.ReadFile(srtCachePath); err == nil {
		fmt.Fprintf(os.Stderr, "  (using cached ASR result: %s)\n", srtCachePath)
		return ParseSRT(string(data), cfg.Language)
	}

	// Cache miss: run whisper and capture SRT
	tmpDir, err := os.MkdirTemp("", "met-asr-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputBase := filepath.Join(tmpDir, "output")
	args := buildWhisperArgs(cfg, wavPath, outputBase)

	cmd := exec.Command(cfg.BinPath, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("whisper-cli failed: %w", err)
	}

	srtPath := outputBase + ".srt"
	srtData, err := os.ReadFile(srtPath)
	if err != nil {
		return nil, fmt.Errorf("read SRT output: %w", err)
	}

	// Save to cache
	os.WriteFile(srtCachePath, srtData, 0644)

	return ParseSRT(string(srtData), cfg.Language)
}

// Transcribe runs whisper-cli without caching. Kept for compatibility.
func Transcribe(cfg WhisperConfig, wavPath string) ([]types.ASRResult, error) {
	tmpDir, err := os.MkdirTemp("", "met-asr-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputBase := filepath.Join(tmpDir, "output")
	args := buildWhisperArgs(cfg, wavPath, outputBase)

	cmd := exec.Command(cfg.BinPath, args...)
	cmd.Stdout = os.Stderr
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
