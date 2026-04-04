package models

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/embedded"
)

// manifestFile is the name of the JSON file that tracks downloaded models.
const manifestFile = ".manifest.json"

// manifestEntry holds metadata for a cached model file.
type manifestEntry struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// ModelsDir returns the directory used to store downloaded models.
// It is derived from embedded.CacheDir() + "/models/".
func ModelsDir() string {
	return filepath.Join(embedded.CacheDir(), "models")
}

// EnsureModel checks if the named model is present on disk; downloads it if not.
// Returns the absolute path to the local model file.
func EnsureModel(name string) (string, error) {
	info, ok := Registry[name]
	if !ok {
		return "", fmt.Errorf("unknown model %q: not found in registry", name)
	}

	dir := ModelsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	dest := filepath.Join(dir, modelFilename(name, info.URL))
	manifestPath := filepath.Join(dir, manifestFile)

	manifest, err := loadManifest(manifestPath)
	if err != nil {
		return "", fmt.Errorf("load manifest: %w", err)
	}

	// Cache hit: file on disk and recorded in manifest.
	if _, statErr := os.Stat(dest); statErr == nil {
		if entry, cached := manifest[name]; cached {
			// If a SHA256 is set in the registry, verify it matches what we recorded.
			if info.SHA256 == "" || entry.SHA256 == info.SHA256 {
				return dest, nil
			}
			// Hash mismatch — fall through to re-download.
			fmt.Fprintf(os.Stderr, "warning: cached %s has wrong hash, re-downloading\n", name)
		}
	}

	// Dispatch to archive handler for .tar.bz2 models.
	if info.IsArchive {
		return ensureArchiveModel(name, info, dir, dest, manifest, manifestPath)
	}

	// Download the model.
	fmt.Fprintf(os.Stderr, "Downloading %s (%s)...\n", name, formatSize(info.Size))
	if err := downloadFile(dest, info.URL, info.Size); err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}

	// Compute and optionally verify SHA-256.
	hash, err := fileHash(dest)
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", name, err)
	}
	if info.SHA256 != "" && hash != info.SHA256 {
		_ = os.Remove(dest)
		return "", fmt.Errorf("SHA-256 mismatch for %s: got %s want %s", name, hash, info.SHA256)
	}

	// Update manifest.
	manifest[name] = manifestEntry{SHA256: hash, Size: info.Size}
	if err := saveManifest(manifestPath, manifest); err != nil {
		return "", fmt.Errorf("save manifest: %w", err)
	}

	return dest, nil
}

// downloadFile fetches url and writes it to dest, printing a simple progress indicator.
func downloadFile(dest, url string, expectedSize int64) error {
	resp, err := http.Get(url) //nolint:gosec // URL comes from the hardcoded registry
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	var written int64
	buf := make([]byte, 1<<20) // 1 MiB buffer
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			written += int64(n)
			if expectedSize > 0 {
				pct := float64(written) / float64(expectedSize) * 100
				fmt.Fprintf(os.Stderr, "\r  %.1f%% (%s / %s)", pct, formatSize(written), formatSize(expectedSize))
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	fmt.Fprintln(os.Stderr) // newline after progress
	return nil
}

// fileHash returns the lowercase hex-encoded SHA-256 digest of the file at path.
func fileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// formatSize formats a byte count as a human-readable string (e.g. "2.0 MB").
func formatSize(bytes int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// loadManifest reads the manifest JSON at path and returns its entries.
// Returns an empty map if the file does not exist.
func loadManifest(path string) (map[string]manifestEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]manifestEntry{}, nil
		}
		return nil, err
	}
	var m map[string]manifestEntry
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// ensureArchiveModel downloads a .tar.bz2 archive, extracts it, renames the
// extracted directory to destDir, and updates the manifest.
func ensureArchiveModel(name string, info ModelInfo, modelsDir, destDir string, manifest map[string]manifestEntry, manifestPath string) (string, error) {
	fmt.Fprintf(os.Stderr, "Downloading %s (%s)...\n", name, formatSize(info.Size))
	tmpFile := filepath.Join(modelsDir, name+".tar.bz2.tmp")
	if err := downloadFile(tmpFile, info.URL, info.Size); err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}
	defer os.Remove(tmpFile)

	cmd := exec.Command("tar", "xjf", tmpFile, "-C", modelsDir)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("extract %s: %w\nstderr: %s", name, err, stderr.String())
	}

	// Find extracted directory and rename to destDir.
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return "", fmt.Errorf("read models dir: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() && strings.Contains(entry.Name(), "sherpa-onnx-sense-voice") {
			extractedDir := filepath.Join(modelsDir, entry.Name())
			if extractedDir != destDir {
				os.RemoveAll(destDir)
				if err := os.Rename(extractedDir, destDir); err != nil {
					return "", fmt.Errorf("rename %s to %s: %w", extractedDir, destDir, err)
				}
			}
			break
		}
	}

	manifest[name] = manifestEntry{SHA256: "", Size: info.Size}
	if err := saveManifest(manifestPath, manifest); err != nil {
		return "", fmt.Errorf("save manifest: %w", err)
	}
	return destDir, nil
}

// saveManifest writes the manifest map to path as indented JSON.
func saveManifest(path string, m map[string]manifestEntry) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
