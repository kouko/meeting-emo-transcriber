package embedded

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestComputeSHA256 verifies the known SHA-256 of "hello world" (no trailing newline).
// Reference: printf '%s' 'hello world' | shasum -a 256
func TestComputeSHA256(t *testing.T) {
	got, err := computeSHA256([]byte("hello world"))
	if err != nil {
		t.Fatalf("computeSHA256 returned error: %v", err)
	}
	const want = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	if got != want {
		t.Errorf("computeSHA256(%q) = %q, want %q", "hello world", got, want)
	}
}

// TestExtractBinary verifies first extraction writes file with correct permissions,
// second call with same hash skips extraction.
func TestExtractBinary(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		t.Fatal(err)
	}

	data := []byte("fake-binary-content")
	destPath := filepath.Join(binDir, "test-bin")

	hash, err := computeSHA256(data)
	if err != nil {
		t.Fatal(err)
	}

	versions := map[string]string{}

	// First extraction: file should be written.
	wrote, err := extractBinary(data, destPath, hash, versions)
	if err != nil {
		t.Fatalf("first extractBinary error: %v", err)
	}
	if !wrote {
		t.Error("first extractBinary should have written the file (wrote=true)")
	}

	// Verify file contents.
	got, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatalf("ReadFile after extract: %v", err)
	}
	if string(got) != string(data) {
		t.Errorf("file content = %q, want %q", got, data)
	}

	// Verify executable permission bit.
	info, err := os.Stat(destPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&0111 == 0 {
		t.Errorf("file mode %v has no execute bit", info.Mode())
	}

	// Verify versions map was updated.
	if versions[destPath] != hash {
		t.Errorf("versions[%q] = %q, want %q", destPath, versions[destPath], hash)
	}

	// Second extraction with same hash: should skip (return false).
	wrote2, err := extractBinary(data, destPath, hash, versions)
	if err != nil {
		t.Fatalf("second extractBinary error: %v", err)
	}
	if wrote2 {
		t.Error("second extractBinary with same hash should skip (return false)")
	}
}

// TestVersionsJSON verifies saveVersions/loadVersions round-trip,
// and that a non-existent file returns an empty map (not an error).
func TestVersionsJSON(t *testing.T) {
	dir := t.TempDir()
	versionsPath := filepath.Join(dir, ".versions.json")

	// Non-existent file should return empty map, not error.
	versions, err := loadVersions(versionsPath)
	if err != nil {
		t.Fatalf("loadVersions on missing file: %v", err)
	}
	if len(versions) != 0 {
		t.Errorf("loadVersions on missing file returned non-empty map: %v", versions)
	}

	// Write some data and read back.
	want := map[string]string{
		"/path/to/bin/whisper-cli":         "abc123",
		"/path/to/bin/ffmpeg":              "def456",
		"/path/to/bin/libonnxruntime.dylib": "ghi789",
	}
	if err := saveVersions(versionsPath, want); err != nil {
		t.Fatalf("saveVersions: %v", err)
	}

	got, err := loadVersions(versionsPath)
	if err != nil {
		t.Fatalf("loadVersions after save: %v", err)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("versions[%q] = %q, want %q", k, got[k], v)
		}
	}

	// Verify the file on disk is valid JSON.
	raw, _ := os.ReadFile(versionsPath)
	var check map[string]string
	if err := json.Unmarshal(raw, &check); err != nil {
		t.Errorf("saved .versions.json is not valid JSON: %v", err)
	}
}

// TestExtractAllCreatesDirectory uses the MET_CACHE_DIR env override to redirect
// the cache dir, then verifies bin/ directory, .versions.json, and non-empty BinPaths.
func TestExtractAllCreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MET_CACHE_DIR", dir)

	paths, err := ExtractAll()
	if err != nil {
		t.Fatalf("ExtractAll: %v", err)
	}

	// Verify bin/ directory was created.
	binDir := filepath.Join(dir, "bin")
	info, err := os.Stat(binDir)
	if err != nil {
		t.Fatalf("bin/ directory not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("bin/ is not a directory")
	}

	// Verify .versions.json was written.
	versionsPath := filepath.Join(dir, ".versions.json")
	if _, err := os.Stat(versionsPath); err != nil {
		t.Fatalf(".versions.json not created: %v", err)
	}

	// Verify BinPaths fields are non-empty.
	if paths.WhisperCLI == "" {
		t.Error("BinPaths.WhisperCLI is empty")
	}
	if paths.FFmpeg == "" {
		t.Error("BinPaths.FFmpeg is empty")
	}
	if paths.ONNXRuntime == "" {
		t.Error("BinPaths.ONNXRuntime is empty")
	}

	// Paths must reside inside bin/.
	for _, p := range []string{paths.WhisperCLI, paths.FFmpeg, paths.ONNXRuntime} {
		rel, err := filepath.Rel(binDir, p)
		if err != nil || rel == "" {
			t.Errorf("path %q is not under binDir %q", p, binDir)
		}
	}

	// Binary files must exist on disk.
	for _, p := range []string{paths.WhisperCLI, paths.FFmpeg, paths.ONNXRuntime} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("binary file %q does not exist: %v", p, err)
		}
	}

	fmt.Printf("BinPaths: %+v\n", paths)
}
