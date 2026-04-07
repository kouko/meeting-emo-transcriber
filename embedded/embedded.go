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
	WhisperCLI    string
	FFmpeg        string
	Diarize       string
	Denoise       string
	SherpaSidecar string
}

// sherpaDylibs is the set of sherpa-onnx dylibs that must be extracted
// alongside the sherpa sidecar binary. The sidecar's LC_RPATH is
// @executable_path/. so these need to land in the same directory as
// metr-sherpa. See scripts/build-sherpa-sidecar.sh for how they get
// embedded in the first place.
//
// The unversioned libonnxruntime.dylib is intentionally not included:
// upstream ships a duplicate file (not a symlink), the sherpa C++ API
// references the versioned filename directly, and we save ~35 MB of
// embedded payload by dropping it.
var sherpaDylibs = []string{
	"libsherpa-onnx-c-api.dylib",
	// The versioned onnxruntime dylib. Extracted via glob so bumping
	// sherpa-onnx-go-macos doesn't require a code change here.
	"libonnxruntime.*.dylib",
}

// MetrDirName is the resource directory name inside the speakers folder.
const MetrDirName = "_metr"

// speakersDir is set by SetSpeakersDir to enable portable mode detection.
var speakersDir string

// SetSpeakersDir sets the speakers directory for portable mode detection.
// Must be called before CacheDir() or ExtractAll().
func SetSpeakersDir(dir string) {
	speakersDir = dir
}

// CacheDir returns the root cache directory.
// Priority: MET_CACHE_DIR env > speakers/_metr/ (portable) > ~/.metr (default)
func CacheDir() string {
	if dir := os.Getenv("MET_CACHE_DIR"); dir != "" {
		return dir
	}
	// Check for portable mode: speakers/_metr/ exists
	if speakersDir != "" {
		portableDir := filepath.Join(speakersDir, MetrDirName)
		if info, err := os.Stat(portableDir); err == nil && info.IsDir() {
			return portableDir
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".metr")
}

// DefaultCacheDir returns ~/.metr (ignoring portable mode). Used by pack/unpack.
func DefaultCacheDir() string {
	if dir := os.Getenv("MET_CACHE_DIR"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".metr")
}

// ExtractAll extracts all embedded binaries to CacheDir()/bin/ and returns
// their paths. Skips extraction if the binary already exists with matching content.
//
// For the sherpa sidecar, the dylibs are also extracted into the same
// directory so that the sidecar's rpath (`@executable_path/.`) resolves
// them at spawn time. All other tools are self-contained binaries with
// no external dylib dependencies.
func ExtractAll() (BinPaths, error) {
	destDir := filepath.Join(CacheDir(), "bin")
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return BinPaths{}, fmt.Errorf("create cache dir: %w", err)
	}

	platform := fmt.Sprintf("%s-%s", runtime.GOOS, runtime.GOARCH)
	tools := map[string]string{
		"whisper-cli":  "",
		"ffmpeg":       "",
		"metr-diarize": "",
		"metr-denoise": "",
		"metr-sherpa":  "",
	}

	for name := range tools {
		srcPath := filepath.Join("bin", platform, name)
		destPath := filepath.Join(destDir, name)
		if err := extractBinary(srcPath, destPath); err != nil {
			return BinPaths{}, fmt.Errorf("extract %s: %w", name, err)
		}
		tools[name] = destPath
	}

	// Extract sherpa-onnx dylibs next to metr-sherpa. They are glob
	// patterns because upstream ships the onnxruntime dylib with a
	// version-suffixed name that bumps on every sherpa-onnx upgrade; we
	// match whatever's in the embedded FS rather than hardcoding a
	// version.
	if err := extractSherpaDylibs(platform, destDir); err != nil {
		return BinPaths{}, fmt.Errorf("extract sherpa dylibs: %w", err)
	}

	return BinPaths{
		WhisperCLI:    tools["whisper-cli"],
		FFmpeg:        tools["ffmpeg"],
		Diarize:       tools["metr-diarize"],
		Denoise:       tools["metr-denoise"],
		SherpaSidecar: tools["metr-sherpa"],
	}, nil
}

// extractSherpaDylibs walks the embedded bin/<platform>/ directory and
// extracts any file whose name matches one of the sherpaDylibs patterns.
// Unlike binaries, dylibs are written 0644 (no execute bit).
func extractSherpaDylibs(platform, destDir string) error {
	entries, err := binFS.ReadDir(filepath.Join("bin", platform))
	if err != nil {
		// No embedded platform dir means dev mode; sherpa sidecar calls
		// will fail later with a clearer "binary not found" error.
		return nil //nolint:nilerr // dev-mode: fall through and let caller surface it
	}
	for _, entry := range entries {
		name := entry.Name()
		if !matchesAnyPattern(name, sherpaDylibs) {
			continue
		}
		srcPath := filepath.Join("bin", platform, name)
		destPath := filepath.Join(destDir, name)
		if err := extractDylib(srcPath, destPath); err != nil {
			return fmt.Errorf("extract %s: %w", name, err)
		}
	}
	return nil
}

func matchesAnyPattern(name string, patterns []string) bool {
	for _, p := range patterns {
		if ok, _ := filepath.Match(p, name); ok {
			return true
		}
	}
	return false
}

// extractDylib is a variant of extractBinary for shared libraries. It
// writes 0644 instead of 0755 (dylibs don't need execute) and otherwise
// follows the same cache-via-sha256 logic.
func extractDylib(srcPath, destPath string) error {
	srcData, err := binFS.ReadFile(srcPath)
	if err != nil {
		return fmt.Errorf("read embedded %s: %w", srcPath, err)
	}

	srcHash := sha256.Sum256(srcData)
	if existing, err := os.ReadFile(destPath); err == nil {
		if sha256.Sum256(existing) == srcHash {
			return nil // cache hit
		}
	}
	if err := os.WriteFile(destPath, srcData, 0644); err != nil {
		return fmt.Errorf("write %s: %w", destPath, err)
	}
	return nil
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
