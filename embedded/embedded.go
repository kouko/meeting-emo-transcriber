// Package embedded handles extraction of bundled binaries to a local cache directory.
// Binaries are embedded via embed.FS from the bin/ subdirectory.
// If binaries are not present (dev mode), ReadFile returns a clear error
// prompting the developer to run 'make deps'.
package embedded

import (
	"crypto/sha256"
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

//go:embed all:bin
var binFS embed.FS

// BinPaths holds absolute paths to the extracted binaries.
type BinPaths struct {
	WhisperCLI string
	FFmpeg     string
	Diarize    string
	Denoise    string
}

// CacheDir returns the root cache directory for extracted binaries.
// The MET_CACHE_DIR environment variable overrides the default
// (~/.metr) to simplify testing.
func CacheDir() string {
	if dir := os.Getenv("MET_CACHE_DIR"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".metr")
}

// ExtractAll extracts all embedded binaries to CacheDir()/bin/ and returns
// their paths. Skips extraction if the binary already exists with matching content.
func ExtractAll() (BinPaths, error) {
	destDir := filepath.Join(CacheDir(), "bin")
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return BinPaths{}, fmt.Errorf("create cache dir: %w", err)
	}

	platform := fmt.Sprintf("%s-%s", runtime.GOOS, runtime.GOARCH)
	tools := map[string]string{
		"whisper-cli":   "",
		"ffmpeg":        "",
		"metr-diarize":  "",
		"metr-denoise":  "",
	}

	for name := range tools {
		srcPath := filepath.Join("bin", platform, name)
		destPath := filepath.Join(destDir, name)
		if err := extractBinary(srcPath, destPath); err != nil {
			return BinPaths{}, fmt.Errorf("extract %s: %w", name, err)
		}
		tools[name] = destPath
	}

	return BinPaths{
		WhisperCLI: tools["whisper-cli"],
		FFmpeg:     tools["ffmpeg"],
		Diarize:    tools["metr-diarize"],
		Denoise:    tools["metr-denoise"],
	}, nil
}

// extractBinary reads a file from the embedded FS and writes it to destPath.
// Skips writing if the destination already has matching SHA-256 content.
func extractBinary(srcPath, destPath string) error {
	srcData, err := binFS.ReadFile(srcPath)
	if err != nil {
		return fmt.Errorf("read embedded %s: %w (run 'make deps' first)", srcPath, err)
	}

	srcHash := sha256.Sum256(srcData)

	if existing, err := os.ReadFile(destPath); err == nil {
		if sha256.Sum256(existing) == srcHash {
			return nil // cache hit
		}
	}

	if err := os.WriteFile(destPath, srcData, 0755); err != nil {
		return fmt.Errorf("write %s: %w", destPath, err)
	}
	return nil
}
