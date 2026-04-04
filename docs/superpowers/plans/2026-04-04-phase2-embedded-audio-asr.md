# Phase 2: Embedded Binaries + Audio + ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `transcribe` command produce real ASR transcripts by embedding external binaries (whisper-cli, ffmpeg, libonnxruntime.dylib), converting audio to 16kHz WAV, and running whisper-cli for speech recognition.

**Architecture:** External binaries are embedded via `go:embed` and extracted to `~/.meeting-emo-transcriber/bin/` with SHA-256 hash-based caching. Audio is converted via ffmpeg subprocess. ASR runs whisper-cli subprocess with Silero VAD, producing SRT output that gets parsed into `[]types.ASRResult`. The transcribe command wires these together with Phase 1's output formatters.

**Tech Stack:** Go 1.25, go:embed, os/exec, go-audio/wav, whisper.cpp v1.7.3, ffmpeg 7.1, ONNX Runtime 1.19.0

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `scripts/build-whisper.sh` | Compile whisper-cli from source with Metal GPU |
| `scripts/download-ffmpeg.sh` | Download prebuilt ffmpeg static binary for macOS arm64 |
| `scripts/download-onnxruntime.sh` | Download libonnxruntime.dylib from GitHub releases |
| `scripts/prepare-all.sh` | Run all three scripts + verify outputs |
| `embedded/binaries/.gitkeep` | Placeholder for git-ignored binary directory |
| `internal/embedded/embedded.go` | go:embed declarations + ExtractAll() + CacheDir() |
| `internal/embedded/embedded_test.go` | Tests for extraction, permissions, hash caching |
| `internal/audio/convert.go` | ffmpeg format conversion + format detection |
| `internal/audio/wav.go` | WAV read/write/segment extraction using go-audio/wav |
| `internal/audio/audio_test.go` | Tests for conversion, WAV I/O, segment extraction |
| `internal/asr/parser.go` | SRT parsing → []types.ASRResult |
| `internal/asr/whisper.go` | whisper-cli subprocess invocation |
| `internal/asr/asr_test.go` | SRT parser unit tests + whisper integration test |
| `internal/models/registry.go` | Model metadata (URLs, SHA-256, sizes) |
| `internal/models/manager.go` | Download, verify, cache model files |
| `internal/models/models_test.go` | ResolveASRModel tests |

### Modified files

| File | Changes |
|------|---------|
| `.gitignore` | Already has `embedded/binaries/` — no change needed |
| `cmd/commands/transcribe.go` | Replace stub with full ASR pipeline |

---

## Task 1: Build Scripts

**Files:**
- Create: `scripts/build-whisper.sh`
- Create: `scripts/download-ffmpeg.sh`
- Create: `scripts/download-onnxruntime.sh`
- Create: `scripts/prepare-all.sh`
- Create: `embedded/binaries/.gitkeep`

- [ ] **Step 1: Create embedded/binaries directory structure**

```bash
mkdir -p embedded/binaries/darwin-arm64
touch embedded/binaries/.gitkeep
```

- [ ] **Step 2: Write scripts/build-whisper.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

WHISPER_VERSION="v1.7.3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/binaries/darwin-arm64"
BUILD_DIR="$PROJECT_ROOT/.build/whisper.cpp"

echo "=== Building whisper-cli $WHISPER_VERSION ==="

# Clone or update
if [ -d "$BUILD_DIR" ]; then
    echo "Using existing whisper.cpp at $BUILD_DIR"
    cd "$BUILD_DIR"
    git fetch --tags
else
    echo "Cloning whisper.cpp..."
    mkdir -p "$(dirname "$BUILD_DIR")"
    git clone https://github.com/ggml-org/whisper.cpp.git "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

git checkout "$WHISPER_VERSION"

# Build with Metal GPU support
echo "Building with Metal GPU support..."
cmake -B build \
    -DGGML_METAL=1 \
    -DGGML_METAL_EMBED_LIBRARY=1 \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"

# Copy binary
mkdir -p "$OUTPUT_DIR"
cp build/bin/whisper-cli "$OUTPUT_DIR/whisper-cli"
chmod 755 "$OUTPUT_DIR/whisper-cli"

echo "whisper-cli built: $OUTPUT_DIR/whisper-cli"
ls -lh "$OUTPUT_DIR/whisper-cli"

# Download Silero VAD model
VAD_MODEL="$OUTPUT_DIR/ggml-silero-v6.2.0.bin"
if [ ! -f "$VAD_MODEL" ]; then
    echo "Downloading Silero VAD model..."
    bash "$BUILD_DIR/models/download-vad-model.sh"
    cp "$BUILD_DIR/models/ggml-silero-v6.2.0.bin" "$VAD_MODEL"
fi
echo "VAD model: $VAD_MODEL"
```

- [ ] **Step 3: Write scripts/download-ffmpeg.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/binaries/darwin-arm64"
OUTPUT_FILE="$OUTPUT_DIR/ffmpeg"

echo "=== Downloading ffmpeg ==="

if [ -f "$OUTPUT_FILE" ]; then
    echo "ffmpeg already exists at $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

# Check if ffmpeg is available via Homebrew
BREW_FFMPEG="$(brew --prefix ffmpeg 2>/dev/null || true)/bin/ffmpeg"
if [ -f "$BREW_FFMPEG" ]; then
    echo "Copying ffmpeg from Homebrew: $BREW_FFMPEG"
    cp "$BREW_FFMPEG" "$OUTPUT_FILE"
    chmod 755 "$OUTPUT_FILE"
    echo "ffmpeg copied: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

# Fallback: download from evermeet.cx (macOS static builds)
echo "Homebrew ffmpeg not found. Please install ffmpeg:"
echo "  brew install ffmpeg"
exit 1
```

- [ ] **Step 4: Write scripts/download-onnxruntime.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

ONNX_VERSION="1.19.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/binaries/darwin-arm64"
OUTPUT_FILE="$OUTPUT_DIR/libonnxruntime.dylib"
TMP_DIR="$PROJECT_ROOT/.build/onnxruntime"

echo "=== Downloading ONNX Runtime v$ONNX_VERSION ==="

if [ -f "$OUTPUT_FILE" ]; then
    echo "libonnxruntime.dylib already exists at $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
    exit 0
fi

mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

TARBALL="onnxruntime-osx-arm64-$ONNX_VERSION.tgz"
URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/$TARBALL"

if [ ! -f "$TMP_DIR/$TARBALL" ]; then
    echo "Downloading from $URL..."
    curl -L -o "$TMP_DIR/$TARBALL" "$URL"
fi

echo "Extracting..."
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

# Find and copy the dylib
DYLIB=$(find "$TMP_DIR" -name "libonnxruntime.$ONNX_VERSION.dylib" -type f | head -1)
if [ -z "$DYLIB" ]; then
    DYLIB=$(find "$TMP_DIR" -name "libonnxruntime.dylib" -type f | head -1)
fi

if [ -z "$DYLIB" ]; then
    echo "ERROR: Could not find libonnxruntime.dylib in extracted archive"
    exit 1
fi

cp "$DYLIB" "$OUTPUT_FILE"
chmod 644 "$OUTPUT_FILE"

echo "libonnxruntime.dylib installed: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
```

- [ ] **Step 5: Write scripts/prepare-all.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/embedded/binaries/darwin-arm64"

echo "=========================================="
echo "  Preparing all embedded binaries"
echo "=========================================="
echo ""

bash "$SCRIPT_DIR/build-whisper.sh"
echo ""
bash "$SCRIPT_DIR/download-ffmpeg.sh"
echo ""
bash "$SCRIPT_DIR/download-onnxruntime.sh"
echo ""

echo "=========================================="
echo "  Verification"
echo "=========================================="

PASS=true
for f in whisper-cli ffmpeg libonnxruntime.dylib; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        SIZE=$(ls -lh "$OUTPUT_DIR/$f" | awk '{print $5}')
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ $f MISSING"
        PASS=false
    fi
done

# Check VAD model
if [ -f "$OUTPUT_DIR/ggml-silero-v6.2.0.bin" ]; then
    SIZE=$(ls -lh "$OUTPUT_DIR/ggml-silero-v6.2.0.bin" | awk '{print $5}')
    echo "  ✓ ggml-silero-v6.2.0.bin ($SIZE)"
else
    echo "  ✗ ggml-silero-v6.2.0.bin MISSING"
    PASS=false
fi

echo ""
if [ "$PASS" = true ]; then
    echo "All binaries ready. You can now build with:"
    echo "  go build -tags embed -o meeting-emo-transcriber ./cmd/main.go"
else
    echo "ERROR: Some binaries are missing. Fix the errors above and re-run."
    exit 1
fi
```

- [ ] **Step 6: Make scripts executable and commit**

```bash
chmod +x scripts/build-whisper.sh scripts/download-ffmpeg.sh scripts/download-onnxruntime.sh scripts/prepare-all.sh
git add scripts/ embedded/binaries/.gitkeep
git commit -m "feat: build scripts for whisper-cli, ffmpeg, onnxruntime binaries"
```

---

## Task 2: Embedded Binary Extraction

**Files:**
- Create: `internal/embedded/embedded.go`
- Create: `internal/embedded/embedded_test.go`

- [ ] **Step 1: Write the failing test for hash computation and extraction**

Create `internal/embedded/embedded_test.go`:

```go
package embedded

import (
	"os"
	"path/filepath"
	"testing"
)

func TestComputeSHA256(t *testing.T) {
	data := []byte("hello world")
	hash := computeSHA256(data)
	// SHA-256 of "hello world"
	expected := "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	if hash != expected {
		t.Errorf("computeSHA256 = %s, want %s", hash, expected)
	}
}

func TestExtractBinary(t *testing.T) {
	tmpDir := t.TempDir()
	data := []byte("#!/bin/sh\necho hello\n")
	hash := computeSHA256(data)

	path := filepath.Join(tmpDir, "test-bin")

	// First extraction: should write file
	written, err := extractBinary(data, path, 0755, hash, nil)
	if err != nil {
		t.Fatalf("extractBinary: %v", err)
	}
	if !written {
		t.Error("expected first extraction to write file")
	}

	// Verify file exists with correct permissions
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Mode().Perm() != 0755 {
		t.Errorf("permissions = %o, want 0755", info.Mode().Perm())
	}

	// Verify content
	content, _ := os.ReadFile(path)
	if string(content) != string(data) {
		t.Error("extracted content mismatch")
	}

	// Second extraction with same hash: should skip
	versions := map[string]string{path: hash}
	written, err = extractBinary(data, path, 0755, hash, versions)
	if err != nil {
		t.Fatalf("extractBinary (cached): %v", err)
	}
	if written {
		t.Error("expected second extraction to skip (cache hit)")
	}
}

func TestVersionsJSON(t *testing.T) {
	tmpDir := t.TempDir()
	versionsPath := filepath.Join(tmpDir, ".versions.json")

	// Write
	v := map[string]string{"whisper-cli": "abc123", "ffmpeg": "def456"}
	if err := saveVersions(versionsPath, v); err != nil {
		t.Fatalf("saveVersions: %v", err)
	}

	// Read back
	loaded, err := loadVersions(versionsPath)
	if err != nil {
		t.Fatalf("loadVersions: %v", err)
	}
	if loaded["whisper-cli"] != "abc123" || loaded["ffmpeg"] != "def456" {
		t.Errorf("loaded versions mismatch: %v", loaded)
	}

	// Read non-existent file returns empty map
	empty, err := loadVersions(filepath.Join(tmpDir, "nonexistent.json"))
	if err != nil {
		t.Fatalf("loadVersions nonexistent: %v", err)
	}
	if len(empty) != 0 {
		t.Error("expected empty map for nonexistent file")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/embedded/ -v`
Expected: FAIL — package does not exist

- [ ] **Step 3: Write the implementation**

Create `internal/embedded/embedded.go`:

```go
package embedded

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// BinPaths holds paths to all extracted binaries.
type BinPaths struct {
	WhisperCLI  string
	FFmpeg      string
	ONNXRuntime string
}

// CacheDir returns the cache root: ~/.meeting-emo-transcriber
func CacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".meeting-emo-transcriber")
}

// binDir returns the binary cache directory: ~/.meeting-emo-transcriber/bin
func binDir() string {
	return filepath.Join(CacheDir(), "bin")
}

// computeSHA256 returns the hex-encoded SHA-256 hash of data.
func computeSHA256(data []byte) string {
	h := sha256.Sum256(data)
	return fmt.Sprintf("%x", h[:])
}

// loadVersions reads .versions.json from path. Returns empty map if file doesn't exist.
func loadVersions(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]string), nil
		}
		return nil, err
	}
	var v map[string]string
	if err := json.Unmarshal(data, &v); err != nil {
		return make(map[string]string), nil
	}
	return v, nil
}

// saveVersions writes the versions map to path as JSON.
func saveVersions(path string, v map[string]string) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// extractBinary writes data to path with perm if the hash doesn't match versions cache.
// Returns true if the file was written, false if skipped (cache hit).
func extractBinary(data []byte, path string, perm os.FileMode, hash string, versions map[string]string) (bool, error) {
	if versions != nil {
		if cached, ok := versions[path]; ok && cached == hash {
			return false, nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return false, err
	}
	if err := os.WriteFile(path, data, perm); err != nil {
		return false, err
	}
	return true, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/embedded/ -v`
Expected: PASS — all 3 tests green

- [ ] **Step 5: Commit**

```bash
git add internal/embedded/
git commit -m "feat: embedded binary extraction with SHA-256 cache"
```

---

## Task 3: Embedded ExtractAll with go:embed

**Files:**
- Create: `internal/embedded/extract.go`
- Modify: `internal/embedded/embedded.go` (add embed vars behind build tag)

> **Note:** This task creates the `ExtractAll()` function that ties embedded binary data to the extraction logic. During development without real binaries, we use placeholder data. The real `go:embed` directives are activated with `-tags embed` once `scripts/prepare-all.sh` has been run.

- [ ] **Step 1: Write the failing test for ExtractAll**

Add to `internal/embedded/embedded_test.go`:

```go
func TestExtractAllCreatesDirectory(t *testing.T) {
	// Override cache dir for testing
	tmpDir := t.TempDir()
	origCacheDir := os.Getenv("MET_CACHE_DIR")
	os.Setenv("MET_CACHE_DIR", tmpDir)
	defer os.Setenv("MET_CACHE_DIR", origCacheDir)

	paths, err := ExtractAll()
	if err != nil {
		t.Fatalf("ExtractAll: %v", err)
	}

	// Verify paths are under tmpDir
	if paths.WhisperCLI == "" || paths.FFmpeg == "" || paths.ONNXRuntime == "" {
		t.Errorf("expected non-empty paths: %+v", paths)
	}

	// Verify bin directory was created
	binDir := filepath.Join(tmpDir, "bin")
	info, err := os.Stat(binDir)
	if err != nil {
		t.Fatalf("bin dir not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("bin path is not a directory")
	}

	// Verify .versions.json exists
	versionsPath := filepath.Join(binDir, ".versions.json")
	if _, err := os.Stat(versionsPath); err != nil {
		t.Errorf(".versions.json not created: %v", err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/embedded/ -v -run TestExtractAll`
Expected: FAIL — ExtractAll not defined

- [ ] **Step 3: Update CacheDir to support MET_CACHE_DIR env override and write ExtractAll**

Update `internal/embedded/embedded.go` — replace the `CacheDir()` function:

```go
// CacheDir returns the cache root.
// Honors MET_CACHE_DIR env var for testing; defaults to ~/.meeting-emo-transcriber
func CacheDir() string {
	if dir := os.Getenv("MET_CACHE_DIR"); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".meeting-emo-transcriber")
}
```

Create `internal/embedded/extract.go`:

```go
package embedded

import (
	"fmt"
	"os"
	"path/filepath"
)

// binarySpec describes one binary to extract.
type binarySpec struct {
	name string
	data []byte
	perm os.FileMode
}

// ExtractAll extracts all embedded binaries to CacheDir()/bin/.
// Uses SHA-256 hash to skip extraction when cached version matches.
func ExtractAll() (BinPaths, error) {
	dir := binDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return BinPaths{}, fmt.Errorf("create bin dir: %w", err)
	}

	versionsPath := filepath.Join(dir, ".versions.json")
	versions, err := loadVersions(versionsPath)
	if err != nil {
		return BinPaths{}, fmt.Errorf("load versions: %w", err)
	}

	specs := []binarySpec{
		{name: "whisper-cli", data: whisperCLIData(), perm: 0755},
		{name: "ffmpeg", data: ffmpegData(), perm: 0755},
		{name: "libonnxruntime.dylib", data: onnxRuntimeData(), perm: 0644},
	}

	for _, spec := range specs {
		path := filepath.Join(dir, spec.name)
		hash := computeSHA256(spec.data)
		written, err := extractBinary(spec.data, path, spec.perm, hash, versions)
		if err != nil {
			return BinPaths{}, fmt.Errorf("extract %s: %w", spec.name, err)
		}
		if written {
			versions[path] = hash
		}
	}

	if err := saveVersions(versionsPath, versions); err != nil {
		return BinPaths{}, fmt.Errorf("save versions: %w", err)
	}

	return BinPaths{
		WhisperCLI:  filepath.Join(dir, "whisper-cli"),
		FFmpeg:      filepath.Join(dir, "ffmpeg"),
		ONNXRuntime: filepath.Join(dir, "libonnxruntime.dylib"),
	}, nil
}

// Default data functions — return placeholder data for development.
// These are overridden by embed_prod.go when building with -tags embed.

func whisperCLIData() []byte  { return []byte("whisper-cli-placeholder") }
func ffmpegData() []byte      { return []byte("ffmpeg-placeholder") }
func onnxRuntimeData() []byte { return []byte("onnxruntime-placeholder") }
```

- [ ] **Step 4: Create the production embed file**

Create `internal/embedded/embed_prod.go`:

```go
//go:build embed

package embedded

import _ "embed"

//go:embed ../../embedded/binaries/darwin-arm64/whisper-cli
var embeddedWhisperCLI []byte

//go:embed ../../embedded/binaries/darwin-arm64/ffmpeg
var embeddedFFmpeg []byte

//go:embed ../../embedded/binaries/darwin-arm64/libonnxruntime.dylib
var embeddedONNXRuntime []byte

func init() {
	whisperCLIData = func() []byte { return embeddedWhisperCLI }
	ffmpegData = func() []byte { return embeddedFFmpeg }
	onnxRuntimeData = func() []byte { return embeddedONNXRuntime }
}
```

Wait — `init()` overriding package-level functions won't work because they're defined as regular functions, not function variables. We need to change the approach. Update `extract.go` to use function variables instead:

Replace the bottom of `internal/embedded/extract.go` — change the data functions to variables:

```go
// Data providers — default to placeholders for dev mode.
// Overridden by embed_prod.go via init() when building with -tags embed.
var whisperCLIData = func() []byte { return []byte("whisper-cli-placeholder") }
var ffmpegData = func() []byte { return []byte("ffmpeg-placeholder") }
var onnxRuntimeData = func() []byte { return []byte("onnxruntime-placeholder") }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/embedded/ -v`
Expected: PASS — all tests green (using placeholder data)

- [ ] **Step 6: Commit**

```bash
git add internal/embedded/
git commit -m "feat: ExtractAll with go:embed support and dev-mode placeholders"
```

---

## Task 4: SRT Parser

**Files:**
- Create: `internal/asr/parser.go`
- Create: `internal/asr/asr_test.go`

- [ ] **Step 1: Write the failing tests**

Create `internal/asr/asr_test.go`:

```go
package asr

import (
	"testing"
)

func TestParseSRTTimestamp(t *testing.T) {
	tests := []struct {
		input    string
		expected float64
	}{
		{"00:00:00,000", 0.0},
		{"00:00:01,000", 1.0},
		{"00:00:01,500", 1.5},
		{"00:01:00,000", 60.0},
		{"01:00:00,000", 3600.0},
		{"01:23:45,678", 5025.678},
		{"00:00:00,001", 0.001},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseSRTTimestamp(tt.input)
			if err != nil {
				t.Fatalf("ParseSRTTimestamp(%q): %v", tt.input, err)
			}
			diff := got - tt.expected
			if diff < -0.0005 || diff > 0.0005 {
				t.Errorf("ParseSRTTimestamp(%q) = %f, want %f", tt.input, got, tt.expected)
			}
		})
	}
}

func TestParseSRTTimestampInvalid(t *testing.T) {
	invalids := []string{"", "00:00:00", "abc", "00:00:00.000", "1:2:3,4"}
	for _, input := range invalids {
		t.Run(input, func(t *testing.T) {
			_, err := ParseSRTTimestamp(input)
			if err == nil {
				t.Errorf("expected error for %q", input)
			}
		})
	}
}

func TestParseSRT(t *testing.T) {
	input := `1
00:00:00,000 --> 00:00:03,500
Hello, this is a test.

2
00:00:03,500 --> 00:00:07,000
今天的會議開始了。

3
00:00:07,000 --> 00:00:12,500
Let's look at the Q1 data.
`

	results, err := ParseSRT(input, "auto")
	if err != nil {
		t.Fatalf("ParseSRT: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Check first result
	if results[0].Start != 0.0 {
		t.Errorf("result[0].Start = %f, want 0.0", results[0].Start)
	}
	if results[0].End != 3.5 {
		t.Errorf("result[0].End = %f, want 3.5", results[0].End)
	}
	if results[0].Text != "Hello, this is a test." {
		t.Errorf("result[0].Text = %q", results[0].Text)
	}
	if results[0].Language != "auto" {
		t.Errorf("result[0].Language = %q, want \"auto\"", results[0].Language)
	}

	// Check second result
	if results[1].Text != "今天的會議開始了。" {
		t.Errorf("result[1].Text = %q", results[1].Text)
	}
}

func TestParseSRTMultilineText(t *testing.T) {
	input := `1
00:00:00,000 --> 00:00:05,000
First line of subtitle.
Second line of subtitle.

`

	results, err := ParseSRT(input, "en")
	if err != nil {
		t.Fatalf("ParseSRT: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	expected := "First line of subtitle. Second line of subtitle."
	if results[0].Text != expected {
		t.Errorf("Text = %q, want %q", results[0].Text, expected)
	}
}

func TestParseSRTEmpty(t *testing.T) {
	results, err := ParseSRT("", "en")
	if err != nil {
		t.Fatalf("ParseSRT: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestParseSRTWhitespace(t *testing.T) {
	results, err := ParseSRT("   \n\n  \n", "en")
	if err != nil {
		t.Fatalf("ParseSRT: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/asr/ -v`
Expected: FAIL — package does not exist

- [ ] **Step 3: Implement the SRT parser**

Create `internal/asr/parser.go`:

```go
package asr

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// ParseSRTTimestamp parses "HH:MM:SS,mmm" into seconds as float64.
func ParseSRTTimestamp(ts string) (float64, error) {
	// Expected format: HH:MM:SS,mmm (exactly)
	parts := strings.Split(ts, ":")
	if len(parts) != 3 {
		return 0, fmt.Errorf("invalid SRT timestamp: %q", ts)
	}

	h, err := strconv.Atoi(parts[0])
	if err != nil || len(parts[0]) != 2 {
		return 0, fmt.Errorf("invalid hours in SRT timestamp: %q", ts)
	}

	m, err := strconv.Atoi(parts[1])
	if err != nil || len(parts[1]) != 2 {
		return 0, fmt.Errorf("invalid minutes in SRT timestamp: %q", ts)
	}

	secParts := strings.Split(parts[2], ",")
	if len(secParts) != 2 || len(secParts[0]) != 2 || len(secParts[1]) != 3 {
		return 0, fmt.Errorf("invalid seconds in SRT timestamp: %q", ts)
	}

	s, err := strconv.Atoi(secParts[0])
	if err != nil {
		return 0, fmt.Errorf("invalid seconds in SRT timestamp: %q", ts)
	}

	ms, err := strconv.Atoi(secParts[1])
	if err != nil {
		return 0, fmt.Errorf("invalid milliseconds in SRT timestamp: %q", ts)
	}

	return float64(h)*3600 + float64(m)*60 + float64(s) + float64(ms)/1000.0, nil
}

// ParseSRT parses SRT-formatted text into ASRResult slices.
// The language parameter is passed through to each result since
// whisper-cli SRT output does not include language metadata.
func ParseSRT(content string, language string) ([]types.ASRResult, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil, nil
	}

	// Split by double newlines to get blocks
	// Normalize line endings first
	content = strings.ReplaceAll(content, "\r\n", "\n")
	blocks := strings.Split(content, "\n\n")

	var results []types.ASRResult
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		lines := strings.Split(block, "\n")
		if len(lines) < 2 {
			continue
		}

		// Line 0: sequence number (skip)
		// Line 1: timestamp line "HH:MM:SS,mmm --> HH:MM:SS,mmm"
		// Lines 2+: text content
		tsLine := strings.TrimSpace(lines[1])
		tsParts := strings.Split(tsLine, " --> ")
		if len(tsParts) != 2 {
			continue
		}

		start, err := ParseSRTTimestamp(strings.TrimSpace(tsParts[0]))
		if err != nil {
			return nil, fmt.Errorf("parse start timestamp in block: %w", err)
		}

		end, err := ParseSRTTimestamp(strings.TrimSpace(tsParts[1]))
		if err != nil {
			return nil, fmt.Errorf("parse end timestamp in block: %w", err)
		}

		// Join remaining lines as text (multi-line subtitles)
		var textParts []string
		for _, line := range lines[2:] {
			line = strings.TrimSpace(line)
			if line != "" {
				textParts = append(textParts, line)
			}
		}
		text := strings.Join(textParts, " ")

		if text == "" {
			continue
		}

		results = append(results, types.ASRResult{
			Start:    start,
			End:      end,
			Text:     text,
			Language: language,
		})
	}

	return results, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/asr/ -v`
Expected: PASS — all 6 tests green

- [ ] **Step 5: Commit**

```bash
git add internal/asr/
git commit -m "feat: SRT parser for whisper-cli output"
```

---

## Task 5: Audio Format Detection + Conversion

**Files:**
- Create: `internal/audio/convert.go`
- Create: `internal/audio/audio_test.go`

- [ ] **Step 1: Write the failing tests**

Create `internal/audio/audio_test.go`:

```go
package audio

import (
	"strings"
	"testing"
)

func TestParseFFmpegInfo(t *testing.T) {
	// Simulated ffmpeg -i stderr output for a 16kHz mono PCM WAV
	stderrWAV := `ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers
Input #0, wav, from 'test.wav':
  Duration: 00:00:10.00, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s`

	info := parseFFmpegInfo(stderrWAV)
	if info.Codec != "pcm_s16le" {
		t.Errorf("Codec = %q, want pcm_s16le", info.Codec)
	}
	if info.SampleRate != 16000 {
		t.Errorf("SampleRate = %d, want 16000", info.SampleRate)
	}
	if info.Channels != 1 {
		t.Errorf("Channels = %d, want 1", info.Channels)
	}

	// Stereo MP3 at 44100
	stderrMP3 := `Input #0, mp3, from 'test.mp3':
  Duration: 00:03:25.00, bitrate: 320 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s`

	info2 := parseFFmpegInfo(stderrMP3)
	if info2.Codec != "mp3" {
		t.Errorf("Codec = %q, want mp3", info2.Codec)
	}
	if info2.SampleRate != 44100 {
		t.Errorf("SampleRate = %d, want 44100", info2.SampleRate)
	}
	if info2.Channels != 2 {
		t.Errorf("Channels = %d, want 2 (stereo)", info2.Channels)
	}
}

func TestIsTargetFormat(t *testing.T) {
	tests := []struct {
		info     audioInfo
		expected bool
	}{
		{audioInfo{"pcm_s16le", 16000, 1}, true},
		{audioInfo{"pcm_s16le", 44100, 1}, false},  // wrong sample rate
		{audioInfo{"pcm_s16le", 16000, 2}, false},  // wrong channels
		{audioInfo{"mp3", 16000, 1}, false},          // wrong codec
		{audioInfo{"", 0, 0}, false},                 // empty
	}
	for _, tt := range tests {
		got := isTargetFormat(tt.info)
		if got != tt.expected {
			t.Errorf("isTargetFormat(%+v) = %v, want %v", tt.info, got, tt.expected)
		}
	}
}

func TestBuildFFmpegArgs(t *testing.T) {
	args := buildFFmpegArgs("/path/to/ffmpeg", "/input.mp3", "/output.wav")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-i /input.mp3") {
		t.Errorf("missing input flag: %s", joined)
	}
	if !strings.Contains(joined, "-acodec pcm_s16le") {
		t.Errorf("missing codec: %s", joined)
	}
	if !strings.Contains(joined, "-ar 16000") {
		t.Errorf("missing sample rate: %s", joined)
	}
	if !strings.Contains(joined, "-ac 1") {
		t.Errorf("missing channels: %s", joined)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/audio/ -v`
Expected: FAIL — package does not exist

- [ ] **Step 3: Implement audio conversion**

Create `internal/audio/convert.go`:

```go
package audio

import (
	"bytes"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// audioInfo holds detected audio format metadata.
type audioInfo struct {
	Codec      string
	SampleRate int
	Channels   int
}

// channelNames maps ffmpeg channel layout names to channel counts.
var channelNames = map[string]int{
	"mono":   1,
	"stereo": 2,
}

// parseFFmpegInfo extracts codec, sample rate, and channels from ffmpeg -i stderr.
func parseFFmpegInfo(stderr string) audioInfo {
	var info audioInfo

	// Match: "Stream #0:0: Audio: <codec> (...), <rate> Hz, <layout>"
	// Example: "Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s"
	re := regexp.MustCompile(`Stream #\d+:\d+.*Audio:\s*(\w+).*?,\s*(\d+)\s*Hz,\s*(\w+)`)
	matches := re.FindStringSubmatch(stderr)
	if len(matches) < 4 {
		return info
	}

	info.Codec = matches[1]

	if rate, err := strconv.Atoi(matches[2]); err == nil {
		info.SampleRate = rate
	}

	layout := strings.ToLower(matches[3])
	if ch, ok := channelNames[layout]; ok {
		info.Channels = ch
	} else if ch, err := strconv.Atoi(layout); err == nil {
		info.Channels = ch
	}

	return info
}

// isTargetFormat returns true if audio is already 16kHz mono PCM s16le.
func isTargetFormat(info audioInfo) bool {
	return info.Codec == "pcm_s16le" && info.SampleRate == 16000 && info.Channels == 1
}

// DetectFormat runs ffmpeg -i to detect the audio format without converting.
func DetectFormat(ffmpegPath, inputPath string) (audioInfo, error) {
	cmd := exec.Command(ffmpegPath, "-i", inputPath, "-f", "null", "-")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	// ffmpeg -i returns exit code 1 when no output specified, that's expected
	_ = cmd.Run()
	return parseFFmpegInfo(stderr.String()), nil
}

// buildFFmpegArgs constructs the ffmpeg conversion arguments.
func buildFFmpegArgs(ffmpegPath, inputPath, outputPath string) []string {
	return []string{
		"-y",
		"-i", inputPath,
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		outputPath,
	}
}

// ConvertToWAV converts any audio format to 16kHz mono PCM WAV.
// If the input is already in the target format, it copies the file instead.
func ConvertToWAV(ffmpegPath, inputPath, outputPath string) error {
	info, err := DetectFormat(ffmpegPath, inputPath)
	if err != nil {
		return fmt.Errorf("detect format: %w", err)
	}

	if isTargetFormat(info) {
		// Already correct format — copy file
		cmd := exec.Command("cp", inputPath, outputPath)
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("copy file: %w", err)
		}
		return nil
	}

	args := buildFFmpegArgs(ffmpegPath, inputPath, outputPath)
	cmd := exec.Command(ffmpegPath, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg conversion failed: %w\nstderr: %s", err, stderr.String())
	}
	return nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/audio/ -v`
Expected: PASS — all tests green

- [ ] **Step 5: Commit**

```bash
git add internal/audio/
git commit -m "feat: audio format detection and ffmpeg conversion"
```

---

## Task 6: WAV Read/Write/Segment

**Files:**
- Create: `internal/audio/wav.go`
- Modify: `internal/audio/audio_test.go` (add WAV tests)

- [ ] **Step 1: Write the failing tests**

Add to `internal/audio/audio_test.go`:

```go
func TestReadWriteWAV(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test.wav")

	// Create test samples: 1 second of silence at 16kHz
	samples := make([]float32, 16000)
	for i := range samples {
		samples[i] = float32(i) / 16000.0 // ramp signal
	}

	// Write
	if err := WriteWAV(wavPath, samples, 16000); err != nil {
		t.Fatalf("WriteWAV: %v", err)
	}

	// Read back
	readSamples, sampleRate, err := ReadWAV(wavPath)
	if err != nil {
		t.Fatalf("ReadWAV: %v", err)
	}
	if sampleRate != 16000 {
		t.Errorf("sampleRate = %d, want 16000", sampleRate)
	}
	if len(readSamples) != len(samples) {
		t.Fatalf("len = %d, want %d", len(readSamples), len(samples))
	}

	// Check values are approximately equal (int16 quantization loses precision)
	for i := 0; i < 100; i++ {
		diff := readSamples[i] - samples[i]
		if diff > 0.001 || diff < -0.001 {
			t.Errorf("sample[%d] = %f, want ~%f (diff=%f)", i, readSamples[i], samples[i], diff)
			break
		}
	}
}

func TestExtractSegment(t *testing.T) {
	// 2 seconds at 16kHz = 32000 samples
	samples := make([]float32, 32000)
	for i := range samples {
		samples[i] = float32(i)
	}

	// Extract 0.5s to 1.5s
	seg := ExtractSegment(samples, 16000, 0.5, 1.5)
	expectedLen := 16000 // 1 second
	if len(seg) != expectedLen {
		t.Errorf("segment len = %d, want %d", len(seg), expectedLen)
	}
	// First sample should be at offset 8000 (0.5s * 16000)
	if seg[0] != 8000.0 {
		t.Errorf("seg[0] = %f, want 8000.0", seg[0])
	}
}

func TestExtractSegmentBounds(t *testing.T) {
	samples := make([]float32, 16000) // 1 second

	// Request beyond end — should clamp
	seg := ExtractSegment(samples, 16000, 0.5, 2.0)
	expectedLen := 8000 // 0.5s to end (1.0s)
	if len(seg) != expectedLen {
		t.Errorf("segment len = %d, want %d", len(seg), expectedLen)
	}

	// Start beyond end — should return empty
	seg = ExtractSegment(samples, 16000, 2.0, 3.0)
	if len(seg) != 0 {
		t.Errorf("expected empty segment, got len=%d", len(seg))
	}
}
```

Add this import at the top of the test file: `"path/filepath"`

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/audio/ -v -run "TestReadWrite|TestExtract"`
Expected: FAIL — ReadWAV, WriteWAV, ExtractSegment not defined

- [ ] **Step 3: Implement WAV operations**

Create `internal/audio/wav.go`:

```go
package audio

import (
	"fmt"
	"math"
	"os"

	goaudio "github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

// ReadWAV reads a WAV file and returns float32 samples normalized to [-1, 1].
func ReadWAV(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open wav: %w", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	if !dec.IsValidFile() {
		return nil, 0, fmt.Errorf("invalid WAV file: %s", path)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("read PCM: %w", err)
	}

	sampleRate := int(dec.SampleRate)
	bitDepth := int(dec.BitDepth)
	maxVal := float32(math.Pow(2, float64(bitDepth-1)))

	samples := make([]float32, len(buf.Data))
	for i, v := range buf.Data {
		samples[i] = float32(v) / maxVal
	}

	return samples, sampleRate, nil
}

// WriteWAV writes float32 samples to a 16-bit PCM WAV file.
func WriteWAV(path string, samples []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create wav: %w", err)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, 1, 1) // 16-bit, mono, PCM

	buf := &goaudio.IntBuffer{
		Data:           make([]int, len(samples)),
		Format:         &goaudio.Format{SampleRate: sampleRate, NumChannels: 1},
		SourceBitDepth: 16,
	}

	maxVal := float32(math.Pow(2, 15))
	for i, s := range samples {
		// Clamp to [-1, 1]
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		buf.Data[i] = int(s * maxVal)
	}

	if err := enc.Write(buf); err != nil {
		return fmt.Errorf("write pcm: %w", err)
	}

	return enc.Close()
}

// ExtractSegment returns a slice of samples between start and end seconds.
// Clamps to valid bounds.
func ExtractSegment(samples []float32, sampleRate int, start, end float64) []float32 {
	startIdx := int(start * float64(sampleRate))
	endIdx := int(end * float64(sampleRate))

	if startIdx < 0 {
		startIdx = 0
	}
	if endIdx > len(samples) {
		endIdx = len(samples)
	}
	if startIdx >= endIdx {
		return nil
	}

	seg := make([]float32, endIdx-startIdx)
	copy(seg, samples[startIdx:endIdx])
	return seg
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/audio/ -v`
Expected: PASS — all tests green

- [ ] **Step 5: Commit**

```bash
git add internal/audio/
git commit -m "feat: WAV read/write/segment with go-audio/wav"
```

---

## Task 7: Model Registry + Manager

**Files:**
- Create: `internal/models/registry.go`
- Create: `internal/models/manager.go`
- Create: `internal/models/models_test.go`

- [ ] **Step 1: Write the failing tests**

Create `internal/models/models_test.go`:

```go
package models

import (
	"testing"
)

func TestResolveASRModel(t *testing.T) {
	tests := []struct {
		language string
		expected string
	}{
		{"auto", "ggml-large-v3"},
		{"en", "ggml-large-v3"},
		{"zh-TW", "ggml-large-v3"}, // Phase 2: all map to large-v3
		{"zh", "ggml-large-v3"},
		{"ja", "ggml-large-v3"},
		{"unknown", "ggml-large-v3"}, // fallback
	}
	for _, tt := range tests {
		t.Run(tt.language, func(t *testing.T) {
			got := ResolveASRModel(tt.language)
			if got != tt.expected {
				t.Errorf("ResolveASRModel(%q) = %q, want %q", tt.language, got, tt.expected)
			}
		})
	}
}

func TestRegistryContainsRequiredModels(t *testing.T) {
	required := []string{"ggml-large-v3", "silero-vad-v6.2.0"}
	for _, name := range required {
		if _, ok := Registry[name]; !ok {
			t.Errorf("Registry missing required model: %q", name)
		}
	}
}

func TestRegistryFieldsNotEmpty(t *testing.T) {
	for name, info := range Registry {
		if info.URL == "" {
			t.Errorf("Registry[%q].URL is empty", name)
		}
		if info.Category == "" {
			t.Errorf("Registry[%q].Category is empty", name)
		}
		if info.Size <= 0 {
			t.Errorf("Registry[%q].Size = %d, want > 0", name, info.Size)
		}
	}
}

func TestModelsDir(t *testing.T) {
	dir := ModelsDir()
	if dir == "" {
		t.Error("ModelsDir() returned empty string")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/models/ -v`
Expected: FAIL — package does not exist

- [ ] **Step 3: Implement registry**

Create `internal/models/registry.go`:

```go
package models

// ModelInfo describes a downloadable model.
type ModelInfo struct {
	Name     string
	URL      string
	SHA256   string
	Size     int64  // bytes, for progress display
	Category string // "asr" | "vad" | "speaker" | "emotion"
}

// Registry contains all known model definitions.
var Registry = map[string]ModelInfo{
	"ggml-large-v3": {
		Name:     "ggml-large-v3",
		URL:      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
		SHA256:   "", // Populated after first download verification
		Size:     3100000000,
		Category: "asr",
	},
	"silero-vad-v6.2.0": {
		Name:     "silero-vad-v6.2.0",
		URL:      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-silero-v6.2.0.bin",
		SHA256:   "",
		Size:     2000000,
		Category: "vad",
	},
}

// ResolveASRModel returns the model name for a given language.
// Phase 2: all languages use ggml-large-v3.
func ResolveASRModel(language string) string {
	// Future: map zh-TW → breeze, zh → belle, ja → kotoba
	return "ggml-large-v3"
}
```

- [ ] **Step 4: Implement manager**

Create `internal/models/manager.go`:

```go
package models

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/internal/embedded"
)

// ModelsDir returns the models cache directory.
func ModelsDir() string {
	return filepath.Join(embedded.CacheDir(), "models")
}

// EnsureModel checks if a model exists locally; downloads it if not.
// Returns the local file path.
func EnsureModel(name string) (string, error) {
	info, ok := Registry[name]
	if !ok {
		return "", fmt.Errorf("unknown model: %q", name)
	}

	dir := ModelsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	filename := info.Name + filepath.Ext(info.URL)
	if filepath.Ext(info.URL) == "" {
		filename = info.Name + ".bin"
	}
	modelPath := filepath.Join(dir, filename)

	// Check manifest
	manifestPath := filepath.Join(dir, ".manifest.json")
	manifest := loadManifest(manifestPath)

	if _, err := os.Stat(modelPath); err == nil {
		if _, ok := manifest[name]; ok {
			return modelPath, nil // cached
		}
	}

	// Download
	fmt.Printf("Downloading %s (%s)...\n", info.Name, formatSize(info.Size))
	if err := downloadFile(info.URL, modelPath); err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}

	// Verify SHA-256 if known
	if info.SHA256 != "" {
		hash, err := fileHash(modelPath)
		if err != nil {
			return "", fmt.Errorf("hash %s: %w", name, err)
		}
		if hash != info.SHA256 {
			os.Remove(modelPath)
			return "", fmt.Errorf("SHA-256 mismatch for %s: got %s, want %s", name, hash, info.SHA256)
		}
	}

	// Update manifest
	manifest[name] = modelPath
	saveManifest(manifestPath, manifest)

	return modelPath, nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	written, err := io.Copy(f, resp.Body)
	if err != nil {
		return err
	}

	fmt.Printf("Downloaded %s\n", formatSize(written))
	return nil
}

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
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func formatSize(bytes int64) string {
	const (
		MB = 1024 * 1024
		GB = 1024 * MB
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	default:
		return fmt.Sprintf("%d bytes", bytes)
	}
}

func loadManifest(path string) map[string]string {
	data, err := os.ReadFile(path)
	if err != nil {
		return make(map[string]string)
	}
	var m map[string]string
	if err := json.Unmarshal(data, &m); err != nil {
		return make(map[string]string)
	}
	return m
}

func saveManifest(path string, m map[string]string) {
	data, _ := json.MarshalIndent(m, "", "  ")
	os.WriteFile(path, data, 0644)
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/models/ -v`
Expected: PASS — all 4 tests green

- [ ] **Step 6: Commit**

```bash
git add internal/models/
git commit -m "feat: model registry and download manager"
```

---

## Task 8: Whisper CLI Wrapper

**Files:**
- Create: `internal/asr/whisper.go`
- Modify: `internal/asr/asr_test.go` (add integration test)

- [ ] **Step 1: Write the failing test**

Add to `internal/asr/asr_test.go`:

```go
func TestBuildWhisperArgs(t *testing.T) {
	cfg := WhisperConfig{
		BinPath:      "/path/to/whisper-cli",
		ModelPath:    "/path/to/model.bin",
		VADModelPath: "/path/to/vad.bin",
		Language:     "auto",
		Threads:      4,
	}

	args := buildWhisperArgs(cfg, "/tmp/input.wav", "/tmp/output")
	joined := strings.Join(args, " ")

	checks := []string{
		"-m /path/to/model.bin",
		"-f /tmp/input.wav",
		"-l auto",
		"-t 4",
		"-osrt",
		"-of /tmp/output",
		"--no-prints",
		"--vad",
		"-vm /path/to/vad.bin",
	}

	for _, check := range checks {
		if !strings.Contains(joined, check) {
			t.Errorf("missing %q in args: %s", check, joined)
		}
	}
}
```

Add `"strings"` to the imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/asr/ -v -run TestBuildWhisperArgs`
Expected: FAIL — buildWhisperArgs not defined

- [ ] **Step 3: Implement whisper CLI wrapper**

Create `internal/asr/whisper.go`:

```go
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

// WhisperConfig holds configuration for a whisper-cli invocation.
type WhisperConfig struct {
	BinPath      string
	ModelPath    string
	VADModelPath string
	Language     string
	Threads      int
}

// buildWhisperArgs constructs the argument list for whisper-cli.
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
	// Create temp output base path
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

	// Read the generated SRT file
	srtPath := outputBase + ".srt"
	srtData, err := os.ReadFile(srtPath)
	if err != nil {
		return nil, fmt.Errorf("read SRT output: %w", err)
	}

	return ParseSRT(string(srtData), cfg.Language)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./internal/asr/ -v`
Expected: PASS — all tests green

- [ ] **Step 5: Commit**

```bash
git add internal/asr/
git commit -m "feat: whisper-cli subprocess wrapper"
```

---

## Task 9: Transcribe Command Implementation

**Files:**
- Modify: `cmd/commands/transcribe.go`

- [ ] **Step 1: Write the full transcribe command implementation**

Replace the contents of `cmd/commands/transcribe.go`:

```go
package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/asr"
	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/output"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newTranscribeCmd() *cobra.Command {
	var (
		inputPath  string
		outputPath string
		format     string
		language   string
		threshold  float32
		noDiscover bool
	)
	cmd := &cobra.Command{
		Use:   "transcribe",
		Short: "Transcribe a meeting recording",
		RunE: func(cmd *cobra.Command, args []string) error {
			if inputPath == "" {
				return fmt.Errorf("--input is required")
			}

			// Validate input file exists
			if _, err := os.Stat(inputPath); err != nil {
				return fmt.Errorf("input file not found: %w", err)
			}

			// Load config
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}

			// Override config with CLI flags
			if language != "auto" || cfg.Language == "" {
				cfg.Language = language
			}
			if format != "" {
				cfg.Format = format
			}

			// Step 1: Extract embedded binaries
			fmt.Println("Extracting embedded binaries...")
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// Step 2: Ensure ASR model
			asrModelName := models.ResolveASRModel(cfg.Language)
			asrModelPath, err := models.EnsureModel(asrModelName)
			if err != nil {
				return fmt.Errorf("ensure ASR model: %w", err)
			}

			// Step 3: Ensure VAD model
			vadModelPath, err := models.EnsureModel("silero-vad-v6.2.0")
			if err != nil {
				return fmt.Errorf("ensure VAD model: %w", err)
			}

			// Step 4: Convert audio to 16kHz mono WAV
			tmpDir, err := os.MkdirTemp("", "met-transcribe-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			wavPath := filepath.Join(tmpDir, "input.wav")
			fmt.Printf("Converting audio: %s\n", inputPath)
			if err := audio.ConvertToWAV(bins.FFmpeg, inputPath, wavPath); err != nil {
				return fmt.Errorf("convert audio: %w", err)
			}

			// Step 5: Run ASR
			fmt.Println("Running speech recognition...")
			whisperCfg := asr.WhisperConfig{
				BinPath:      bins.WhisperCLI,
				ModelPath:    asrModelPath,
				VADModelPath: vadModelPath,
				Language:     cfg.Language,
				Threads:      cfg.Threads,
			}
			asrResults, err := asr.Transcribe(whisperCfg, wavPath)
			if err != nil {
				return fmt.Errorf("ASR: %w", err)
			}

			if len(asrResults) == 0 {
				fmt.Println("No speech detected.")
				return nil
			}

			// Step 6: Build TranscriptResult
			segments := make([]types.TranscriptSegment, len(asrResults))
			for i, r := range asrResults {
				segments[i] = types.TranscriptSegment{
					Start:      r.Start,
					End:        r.End,
					Speaker:    "Unknown",
					Emotion:    types.EmotionInfo{Label: "Neutral", Display: ""},
					AudioEvent: "Speech",
					Language:   r.Language,
					Text:       r.Text,
					Confidence: types.Confidence{Speaker: 0, Emotion: 0},
				}
			}

			// Calculate duration from last segment
			duration := "0s"
			if len(asrResults) > 0 {
				last := asrResults[len(asrResults)-1]
				dur := time.Duration(last.End * float64(time.Second))
				duration = dur.Truncate(time.Second).String()
			}

			result := types.TranscriptResult{
				Metadata: types.Metadata{
					File:               inputPath,
					Duration:           duration,
					SpeakersDetected:   0,
					SpeakersIdentified: 0,
					Date:               time.Now().Format("2006-01-02 15:04:05"),
				},
				Segments: segments,
			}

			// Step 7: Format and output
			formats := config.ParseFormats(cfg.Format)
			for _, f := range formats {
				var formatter interface {
					Format(types.TranscriptResult) (string, error)
				}
				switch f {
				case "txt":
					formatter = &output.TXTFormatter{}
				case "json":
					formatter = &output.JSONFormatter{}
				case "srt":
					formatter = &output.SRTFormatter{}
				default:
					return fmt.Errorf("unknown format: %s", f)
				}

				formatted, err := formatter.Format(result)
				if err != nil {
					return fmt.Errorf("format %s: %w", f, err)
				}

				outPath := resolveOutputPath(inputPath, outputPath, f)
				if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
					return fmt.Errorf("create output dir: %w", err)
				}
				if err := os.WriteFile(outPath, []byte(formatted), 0644); err != nil {
					return fmt.Errorf("write %s: %w", outPath, err)
				}
				fmt.Printf("Output: %s\n", outPath)
			}

			fmt.Printf("Done. %d segments transcribed.\n", len(segments))
			return nil
		},
	}
	cmd.Flags().StringVar(&inputPath, "input", "", "input audio file path (required)")
	cmd.Flags().StringVar(&outputPath, "output", "", "output file path")
	cmd.Flags().StringVar(&format, "format", "txt", "output format: txt|json|srt|all (comma-separated)")
	cmd.Flags().StringVar(&language, "language", "auto", "language: auto|zh-TW|zh|en|ja")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.6, "speaker similarity threshold")
	cmd.Flags().BoolVar(&noDiscover, "no-discover", false, "disable unknown speaker auto-discovery")
	cmd.MarkFlagRequired("input")
	return cmd
}

// resolveOutputPath determines the output file path.
// If outputPath is set, uses it as base (appending extension for multi-format).
// Otherwise, derives from inputPath.
func resolveOutputPath(inputPath, outputPath, format string) string {
	ext := "." + format
	if outputPath != "" {
		base := strings.TrimSuffix(outputPath, filepath.Ext(outputPath))
		return base + ext
	}
	base := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))
	return base + ext
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go build ./cmd/main.go`
Expected: BUILD SUCCESS

- [ ] **Step 3: Run all existing tests to confirm no regressions**

Run: `cd /Users/kouko/GitHub/meeting-emo-transcriber && go test ./... -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add cmd/commands/transcribe.go
git commit -m "feat: transcribe command with full ASR pipeline"
```

---

## Task 10: Build + E2E Verification

This task verifies the entire pipeline works end-to-end.

- [ ] **Step 1: Run prepare-all.sh to build/download all binaries**

```bash
cd /Users/kouko/GitHub/meeting-emo-transcriber
bash scripts/prepare-all.sh
```

Expected: All 4 items show ✓ (whisper-cli, ffmpeg, libonnxruntime.dylib, ggml-silero-v6.2.0.bin)

- [ ] **Step 2: Build the binary with embed tag**

```bash
go build -tags embed -o meeting-emo-transcriber ./cmd/main.go
```

Expected: BUILD SUCCESS, produces `meeting-emo-transcriber` binary

- [ ] **Step 3: Verify the binary runs**

```bash
./meeting-emo-transcriber --help
./meeting-emo-transcriber transcribe --help
```

Expected: Shows help text with all flags

- [ ] **Step 4: Run E2E test with a real audio file**

```bash
# Generate a short test WAV (1 second of silence — for pipeline smoke test)
ffmpeg -f lavfi -i "sine=frequency=440:duration=3" -acodec pcm_s16le -ar 16000 -ac 1 /tmp/test-tone.wav

# Run transcribe
./meeting-emo-transcriber transcribe --input /tmp/test-tone.wav --format all
```

Expected: Creates `test-tone.txt`, `test-tone.json`, `test-tone.srt` files (content may be empty/minimal for a tone)

- [ ] **Step 5: Run all tests one final time**

```bash
go test ./... -v
```

Expected: ALL PASS

- [ ] **Step 6: Commit any fixes and final verification**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "chore: E2E verification and final fixes for Phase 2"
```

---

## Self-Review Checklist

- **Spec §2 (Embedded Binary):** ✓ Task 1 (scripts), Task 2-3 (extraction + go:embed)
- **Spec §3 (Audio Module):** ✓ Task 5 (convert), Task 6 (WAV I/O)
- **Spec §4 (Model Management):** ✓ Task 7 (registry + manager)
- **Spec §5 (ASR Module):** ✓ Task 4 (SRT parser), Task 8 (whisper wrapper)
- **Spec §6 (CLI Integration):** ✓ Task 9 (transcribe command)
- **Spec §7 (File List):** ✓ All files accounted for in tasks
- **Spec §8 (Verification):** ✓ Task 10 (E2E)
- **No placeholders:** All code blocks are complete
- **Type consistency:** `WhisperConfig`, `BinPaths`, `ModelInfo`, `ASRResult` used consistently
- **TDD approach:** Tests written before implementation in Tasks 2-8
