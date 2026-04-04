// Package embedded handles extraction of bundled binaries to a local cache directory.
// In production (build tag: embed), binaries are read from go:embed byte arrays.
// In dev mode (no build tag), placeholder byte slices are used instead.
package embedded

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// BinPaths holds absolute paths to the extracted binaries.
type BinPaths struct {
	WhisperCLI  string
	FFmpeg      string
	ONNXRuntime string
}

// CacheDir returns the root cache directory for extracted binaries.
// The MET_CACHE_DIR environment variable overrides the default
// (~/.meeting-emo-transcriber) to simplify testing.
func CacheDir() string {
	if dir := os.Getenv("MET_CACHE_DIR"); dir != "" {
		return dir
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".meeting-emo-transcriber")
}

// computeSHA256 returns the lowercase hex-encoded SHA-256 digest of data.
func computeSHA256(data []byte) (string, error) {
	h := sha256.New()
	if _, err := h.Write(data); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// versionsFile is the file name (relative to CacheDir) that stores hash→path mappings.
const versionsFile = ".versions.json"

// loadVersions reads the .versions.json file at path and returns the contents as a map.
// If the file does not exist, an empty map is returned with no error.
func loadVersions(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]string{}, nil
		}
		return nil, err
	}
	var m map[string]string
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// saveVersions writes the versions map to path as indented JSON.
func saveVersions(path string, versions map[string]string) error {
	data, err := json.MarshalIndent(versions, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// extractBinary writes data to destPath with the given permissions
// unless versions[destPath] already equals hash (cache hit).
// Returns (true, nil) when the file was written, (false, nil) on cache hit.
// On write, versions[destPath] is updated to hash.
func extractBinary(data []byte, destPath string, perm os.FileMode, hash string, versions map[string]string) (bool, error) {
	if versions[destPath] == hash {
		// Cache hit: skip extraction.
		return false, nil
	}
	if err := os.WriteFile(destPath, data, perm); err != nil {
		return false, err
	}
	versions[destPath] = hash
	return true, nil
}
