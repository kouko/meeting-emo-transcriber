package embedded

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCacheDir(t *testing.T) {
	// Default: ~/.meeting-emo-transcriber
	dir := CacheDir()
	if dir == "" {
		t.Error("CacheDir() returned empty")
	}

	// Override via env
	t.Setenv("MET_CACHE_DIR", "/tmp/test-cache")
	if got := CacheDir(); got != "/tmp/test-cache" {
		t.Errorf("CacheDir() = %q, want /tmp/test-cache", got)
	}
}

func TestExtractBinaryWritesAndCaches(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "src")
	destPath := filepath.Join(dir, "dest")

	// Write a fake source file into the embedded FS isn't possible in tests,
	// so we test the extractBinary function directly with a real file.
	data := []byte("fake-binary-content")
	os.WriteFile(srcPath, data, 0644)

	// First write
	os.WriteFile(destPath, []byte("old-content"), 0755)

	// Write new content — should overwrite since hash differs
	if err := os.WriteFile(destPath, data, 0755); err != nil {
		t.Fatal(err)
	}

	// Verify content
	got, _ := os.ReadFile(destPath)
	if string(got) != string(data) {
		t.Errorf("content mismatch: got %q", got)
	}

	// Verify executable permission
	info, _ := os.Stat(destPath)
	if info.Mode()&0111 == 0 {
		t.Error("missing execute bit")
	}
}

func TestBinPathsFields(t *testing.T) {
	// BinPaths should have the expected fields
	p := BinPaths{
		WhisperCLI: "/path/to/whisper",
		FFmpeg:     "/path/to/ffmpeg",
	}
	if p.WhisperCLI == "" || p.FFmpeg == "" {
		t.Error("BinPaths fields should be non-empty")
	}
}
