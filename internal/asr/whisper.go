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

const fingerprintChunkSize = 64 * 1024 // 64KB

// contentFingerprint generates a hash from file size + first 64KB + last 64KB.
// This is fast (reads at most 128KB) and content-based, so the same file at
// different paths produces the same fingerprint.
func contentFingerprint(filePath string) (string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	size := info.Size()

	h := sha256.New()
	// Include file size
	fmt.Fprintf(h, "%d|", size)

	// Read first 64KB
	head := make([]byte, fingerprintChunkSize)
	n, _ := f.Read(head)
	h.Write(head[:n])

	// Read last 64KB (if file is large enough to have a distinct tail)
	if size > int64(fingerprintChunkSize) {
		tail := make([]byte, fingerprintChunkSize)
		f.Seek(-int64(fingerprintChunkSize), 2)
		n, _ = f.Read(tail)
		h.Write(tail[:n])
	}

	return hex.EncodeToString(h.Sum(nil)[:16]), nil
}

// cacheKey generates a hash from content fingerprint + language.
// Language determines the model (via ResolveASRModel), so it implicitly covers model changes.
// Prompt is intentionally excluded — it's a hint that doesn't warrant re-running whisper.
func cacheKey(wavPath string, cfg WhisperConfig) (string, error) {
	fp, err := contentFingerprint(wavPath)
	if err != nil {
		return "", err
	}
	key := fmt.Sprintf("%s|%s", fp, cfg.Language)
	h := sha256.Sum256([]byte(key))
	return hex.EncodeToString(h[:16]), nil
}

// TranscribeWithCache checks for a cached SRT result before running whisper-cli.
// On cache hit, returns parsed results immediately (~0s instead of ~7min).
// On cache miss, runs whisper-cli and saves the SRT to cache.
// The cache key is based on file content fingerprint + language + model + prompt,
// so the same audio file at different paths will still hit cache.
func TranscribeWithCache(cfg WhisperConfig, wavPath string) ([]types.ASRResult, error) {
	cacheDir := filepath.Join(embedded.CacheDir(), "cache")
	os.MkdirAll(cacheDir, 0755)

	key, err := cacheKey(wavPath, cfg)
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
