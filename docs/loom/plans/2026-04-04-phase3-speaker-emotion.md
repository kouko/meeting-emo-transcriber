# Phase 3: Speaker Embedding + Emotion Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace placeholder Speaker="Unknown" and Emotion=Neutral in transcribe output with real speaker identification (CAM++) and emotion classification (SenseVoice) via sherpa-onnx Go API.

**Architecture:** sherpa-onnx Go API (`k2-fsa/sherpa-onnx-go-macos`) wraps CAM++ for 512-dim speaker embedding extraction and SenseVoice OfflineRecognizer for emotion+event classification. Both models run in-process via ONNX Runtime bundled in sherpa-onnx. The existing `internal/speaker/matcher.go` handles cosine similarity matching against enrolled profiles.

**Tech Stack:** Go 1.25, sherpa-onnx-go-macos (cgo), CAM++ ONNX, SenseVoice-Small int8 ONNX

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `internal/speaker/extractor.go` | CAM++ embedding extraction via sherpa-onnx SpeakerEmbeddingExtractor |
| `internal/speaker/extractor_test.go` | Tests for extractor (dimension, close) |
| `internal/emotion/classifier.go` | SenseVoice emotion+event classification via sherpa-onnx OfflineRecognizer |
| `internal/emotion/classifier_test.go` | Tests for classifier |

### Modified files

| File | Changes |
|------|---------|
| `go.mod` | Add `github.com/k2-fsa/sherpa-onnx-go-macos` dependency |
| `internal/embedded/embedded.go` | Remove `ONNXRuntime` from BinPaths |
| `internal/embedded/extract.go` | Remove libonnxruntime binSpec and onnxRuntimeData var |
| `internal/embedded/embed_prod.go` | Remove onnxruntime embed directive |
| `internal/embedded/embedded_test.go` | Update tests for 2-binary ExtractAll |
| `internal/models/registry.go` | Add campplus + sensevoice models |
| `internal/models/manager.go` | Support .onnx downloads and .tar.bz2 archive extraction |
| `internal/models/models_test.go` | Add tests for new registry entries |
| `cmd/commands/transcribe.go` | Add speaker/emotion inference to pipeline |
| `cmd/commands/enroll.go` | Real embedding computation |
| `scripts/prepare-all.sh` | Remove onnxruntime verification |

---

## Task 1: Add sherpa-onnx Dependency + Remove libonnxruntime Embed

**Files:**
- Modify: `go.mod`
- Modify: `internal/embedded/embedded.go`
- Modify: `internal/embedded/extract.go`
- Modify: `internal/embedded/embed_prod.go`
- Modify: `internal/embedded/embedded_test.go`
- Modify: `scripts/prepare-all.sh`

- [ ] **Step 1: Add sherpa-onnx-go-macos dependency**

```bash
cd /Users/kouko/GitHub/meeting-emo-transcriber
go get github.com/k2-fsa/sherpa-onnx-go-macos@latest
go mod tidy
```

- [ ] **Step 2: Remove ONNXRuntime from BinPaths**

Edit `internal/embedded/embedded.go` — change BinPaths struct from:

```go
type BinPaths struct {
	WhisperCLI  string
	FFmpeg      string
	ONNXRuntime string
}
```

To:

```go
type BinPaths struct {
	WhisperCLI string
	FFmpeg     string
}
```

- [ ] **Step 3: Remove onnxruntime from extract.go**

Edit `internal/embedded/extract.go`:

Remove the `onnxRuntimeData` var:
```go
// DELETE this line:
var onnxRuntimeData = func() []byte { return []byte("onnxruntime-placeholder") }
```

Remove the onnxruntime binSpec from the `specs` slice:
```go
specs := []binSpec{
	{name: "whisper-cli", perm: 0755, dataFunc: whisperCLIData},
	{name: "ffmpeg", perm: 0755, dataFunc: ffmpegData},
	// DELETE: {name: "libonnxruntime.dylib", perm: 0644, dataFunc: onnxRuntimeData},
}
```

Remove `ONNXRuntime` from the returned BinPaths:
```go
return BinPaths{
	WhisperCLI: paths["whisper-cli"],
	FFmpeg:     paths["ffmpeg"],
	// DELETE: ONNXRuntime: paths["libonnxruntime.dylib"],
}, nil
```

- [ ] **Step 4: Remove onnxruntime from embed_prod.go**

Edit `internal/embedded/embed_prod.go` to remove onnxruntime embed:

```go
//go:build embed

package embedded

import _ "embed"

//go:embed ../../embedded/binaries/darwin-arm64/whisper-cli
var embeddedWhisperCLI []byte

//go:embed ../../embedded/binaries/darwin-arm64/ffmpeg
var embeddedFFmpeg []byte

func init() {
	whisperCLIData = func() []byte { return embeddedWhisperCLI }
	ffmpegData = func() []byte { return embeddedFFmpeg }
}
```

- [ ] **Step 5: Update embedded_test.go**

Edit `internal/embedded/embedded_test.go`:

In `TestVersionsJSON`, remove the onnxruntime entry from the `want` map:
```go
want := map[string]string{
	"/path/to/bin/whisper-cli": "abc123",
	"/path/to/bin/ffmpeg":     "def456",
}
```

In `TestExtractAllCreatesDirectory`, remove ONNXRuntime assertions:
```go
// Remove:
if paths.ONNXRuntime == "" {
	t.Error("BinPaths.ONNXRuntime is empty")
}

// Change the paths slice to only 2 items:
for _, p := range []string{paths.WhisperCLI, paths.FFmpeg} {
```

Both occurrences of `paths.ONNXRuntime` in the path-checking loops must be removed.

- [ ] **Step 6: Update prepare-all.sh**

Edit `scripts/prepare-all.sh`:

Remove the `download-onnxruntime.sh` call:
```bash
# DELETE: bash "$SCRIPT_DIR/download-onnxruntime.sh"
# DELETE: echo ""
```

Remove `libonnxruntime.dylib` from the verification loop:
```bash
for f in whisper-cli ffmpeg; do
```

- [ ] **Step 7: Run tests to verify**

```bash
go test ./internal/embedded/ -v
go build ./cmd/main.go
```

Expected: All embedded tests pass, build succeeds.

- [ ] **Step 8: Commit**

```bash
git add go.mod go.sum internal/embedded/ scripts/prepare-all.sh
git commit -m "refactor: remove libonnxruntime embed, add sherpa-onnx dependency

sherpa-onnx bundles its own ONNX Runtime, so the separately
embedded libonnxruntime.dylib is no longer needed."
```

---

## Task 2: Model Registry + Download Support for ONNX and Archives

**Files:**
- Modify: `internal/models/registry.go`
- Modify: `internal/models/manager.go`
- Modify: `internal/models/models_test.go`

- [ ] **Step 1: Write failing tests for new registry entries**

Add to `internal/models/models_test.go`:

```go
func TestRegistryContainsSpeakerModel(t *testing.T) {
	info, ok := Registry["campplus-sv-zh-cn"]
	if !ok {
		t.Fatal("Registry missing campplus-sv-zh-cn")
	}
	if info.Category != "speaker" {
		t.Errorf("Category = %q, want \"speaker\"", info.Category)
	}
	if info.URL == "" {
		t.Error("URL is empty")
	}
}

func TestRegistryContainsEmotionModel(t *testing.T) {
	info, ok := Registry["sensevoice-small-int8"]
	if !ok {
		t.Fatal("Registry missing sensevoice-small-int8")
	}
	if info.Category != "emotion" {
		t.Errorf("Category = %q, want \"emotion\"", info.Category)
	}
	if info.URL == "" {
		t.Error("URL is empty")
	}
}

func TestModelFilename(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected string
	}{
		{"ggml-large-v3", "https://example.com/model.bin", "ggml-large-v3.bin"},
		{"campplus-sv-zh-cn", "https://example.com/model.onnx", "campplus-sv-zh-cn.onnx"},
		{"sensevoice-small-int8", "https://example.com/archive.tar.bz2", "sensevoice-small-int8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := modelFilename(tt.name, tt.url)
			if got != tt.expected {
				t.Errorf("modelFilename(%q, %q) = %q, want %q", tt.name, tt.url, got, tt.expected)
			}
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/models/ -v -run "TestRegistryContainsSpeaker|TestRegistryContainsEmotion|TestModelFilename"
```

Expected: FAIL — models not in registry, modelFilename not defined.

- [ ] **Step 3: Add models to registry**

Edit `internal/models/registry.go`, add to `Registry` map:

```go
"campplus-sv-zh-cn": {
	Name:     "campplus-sv-zh-cn",
	URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx",
	SHA256:   "",
	Size:     30000000,
	Category: "speaker",
},
"sensevoice-small-int8": {
	Name:     "sensevoice-small-int8",
	URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
	SHA256:   "",
	Size:     228000000,
	Category: "emotion",
},
```

- [ ] **Step 4: Add ModelInfo.IsArchive field and modelFilename helper**

Edit `internal/models/registry.go`:

Add `IsArchive` field to ModelInfo:
```go
type ModelInfo struct {
	Name      string
	URL       string
	SHA256    string
	Size      int64
	Category  string
	IsArchive bool // true for .tar.bz2 archives that need extraction
}
```

Set `IsArchive: true` on the sensevoice entry.

Add the `modelFilename` function:
```go
// modelFilename returns the local filename for a model based on its URL extension.
func modelFilename(name, url string) string {
	switch {
	case strings.HasSuffix(url, ".tar.bz2"):
		return name // directory name for extracted archive
	case strings.HasSuffix(url, ".onnx"):
		return name + ".onnx"
	default:
		return name + ".bin"
	}
}
```

Add `"strings"` to imports.

- [ ] **Step 5: Update EnsureModel to support ONNX and archive downloads**

Edit `internal/models/manager.go`:

Replace the hardcoded `.bin` filename logic in `EnsureModel`:

```go
func EnsureModel(name string) (string, error) {
	info, ok := Registry[name]
	if !ok {
		return "", fmt.Errorf("unknown model %q: not found in registry", name)
	}

	dir := ModelsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	filename := modelFilename(name, info.URL)
	dest := filepath.Join(dir, filename)
	manifestPath := filepath.Join(dir, manifestFile)

	manifest, err := loadManifest(manifestPath)
	if err != nil {
		return "", fmt.Errorf("load manifest: %w", err)
	}

	// Cache hit check
	if _, statErr := os.Stat(dest); statErr == nil {
		if entry, cached := manifest[name]; cached {
			if info.SHA256 == "" || entry.SHA256 == info.SHA256 {
				return dest, nil
			}
			fmt.Fprintf(os.Stderr, "warning: cached %s has wrong hash, re-downloading\n", name)
		}
	}

	if info.IsArchive {
		return ensureArchiveModel(name, info, dir, dest, manifest, manifestPath)
	}

	// Direct download
	fmt.Fprintf(os.Stderr, "Downloading %s (%s)...\n", name, formatSize(info.Size))
	if err := downloadFile(dest, info.URL, info.Size); err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}

	hash, err := fileHash(dest)
	if err != nil {
		return "", fmt.Errorf("hash %s: %w", name, err)
	}
	if info.SHA256 != "" && hash != info.SHA256 {
		_ = os.Remove(dest)
		return "", fmt.Errorf("SHA-256 mismatch for %s: got %s want %s", name, hash, info.SHA256)
	}

	manifest[name] = manifestEntry{SHA256: hash, Size: info.Size}
	if err := saveManifest(manifestPath, manifest); err != nil {
		return "", fmt.Errorf("save manifest: %w", err)
	}

	return dest, nil
}
```

Add the archive download handler:

```go
// ensureArchiveModel downloads a .tar.bz2 archive and extracts it.
func ensureArchiveModel(name string, info ModelInfo, modelsDir, destDir string, manifest map[string]manifestEntry, manifestPath string) (string, error) {
	fmt.Fprintf(os.Stderr, "Downloading %s (%s)...\n", name, formatSize(info.Size))

	// Download to temp file
	tmpFile := filepath.Join(modelsDir, name+".tar.bz2.tmp")
	if err := downloadFile(tmpFile, info.URL, info.Size); err != nil {
		return "", fmt.Errorf("download %s: %w", name, err)
	}
	defer os.Remove(tmpFile)

	// Extract using tar command
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("create dir %s: %w", destDir, err)
	}

	cmd := exec.Command("tar", "xjf", tmpFile, "-C", modelsDir)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("extract %s: %w\nstderr: %s", name, err, stderr.String())
	}

	// The archive extracts to a directory like sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
	// Find the extracted directory and rename to our standard name
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return "", fmt.Errorf("read models dir: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() && strings.Contains(entry.Name(), "sherpa-onnx-sense-voice") {
			extractedDir := filepath.Join(modelsDir, entry.Name())
			// If destDir already exists (from a previous partial extraction), remove it
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
```

Add `"bytes"`, `"os/exec"`, `"strings"` to imports.

- [ ] **Step 6: Run tests**

```bash
go test ./internal/models/ -v
```

Expected: All tests pass including the 3 new ones.

- [ ] **Step 7: Commit**

```bash
git add internal/models/
git commit -m "feat: add CAM++ and SenseVoice to model registry with archive support"
```

---

## Task 3: Speaker Embedding Extractor

**Files:**
- Create: `internal/speaker/extractor.go`
- Create: `internal/speaker/extractor_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/speaker/extractor_test.go`:

```go
package speaker

import (
	"testing"
)

func TestNewExtractorRequiresModel(t *testing.T) {
	// Attempting to create with non-existent model should error
	_, err := NewExtractor("/nonexistent/model.onnx", 1)
	if err == nil {
		t.Error("expected error for non-existent model path")
	}
}

func TestExtractorDim(t *testing.T) {
	// This test requires the actual CAM++ model to be downloaded.
	// Skip in CI / when model not available.
	modelPath := findTestModel("campplus-sv-zh-cn")
	if modelPath == "" {
		t.Skip("CAM++ model not available, skipping")
	}

	ext, err := NewExtractor(modelPath, 1)
	if err != nil {
		t.Fatalf("NewExtractor: %v", err)
	}
	defer ext.Close()

	dim := ext.Dim()
	if dim != 512 {
		t.Errorf("Dim() = %d, want 512", dim)
	}
}

// findTestModel checks if a model is available in the standard models directory.
func findTestModel(name string) string {
	// Try MET_CACHE_DIR first, then default
	paths := []string{
		// Uses the same CacheDir logic as the main app
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
```

Add `"os"` to imports.

Update `findTestModel` to check real paths:

```go
func findTestModel(name string) string {
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".meeting-emo-transcriber", "models", name+".onnx"),
		filepath.Join(home, ".meeting-emo-transcriber", "models", name+".bin"),
		filepath.Join(home, ".meeting-emo-transcriber", "models", name),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
```

Add `"path/filepath"` to imports.

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/speaker/ -v -run TestNewExtractorRequiresModel
```

Expected: FAIL — NewExtractor not defined.

- [ ] **Step 3: Implement extractor**

Create `internal/speaker/extractor.go`:

```go
package speaker

import (
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos/sherpa_onnx"
)

// Extractor wraps sherpa-onnx SpeakerEmbeddingExtractor for CAM++ model.
type Extractor struct {
	inner *sherpa.SpeakerEmbeddingExtractor
	dim   int
}

// NewExtractor creates a speaker embedding extractor using the CAM++ ONNX model.
func NewExtractor(modelPath string, threads int) (*Extractor, error) {
	config := &sherpa.SpeakerEmbeddingExtractorConfig{}
	config.Model = modelPath
	config.NumThreads = threads
	config.Debug = 0
	config.Provider = "cpu"

	inner := sherpa.NewSpeakerEmbeddingExtractor(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create speaker embedding extractor from %s", modelPath)
	}

	dim := inner.Dim()

	return &Extractor{inner: inner, dim: dim}, nil
}

// Extract computes a speaker embedding from audio samples.
// samples must be float32 PCM at the given sampleRate.
// Returns a 512-dimensional float32 embedding vector.
func (e *Extractor) Extract(samples []float32, sampleRate int) ([]float32, error) {
	stream := e.inner.CreateStream()
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)

	if !e.inner.IsReady(stream) {
		return nil, fmt.Errorf("not enough audio data for embedding extraction")
	}

	embedding := e.inner.Compute(stream)
	return embedding, nil
}

// Dim returns the embedding dimension (512 for CAM++).
func (e *Extractor) Dim() int {
	return e.dim
}

// Close releases the underlying sherpa-onnx resources.
func (e *Extractor) Close() {
	if e.inner != nil {
		sherpa.DeleteSpeakerEmbeddingExtractor(e.inner)
		e.inner = nil
	}
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/speaker/ -v
```

Expected: TestNewExtractorRequiresModel PASS (returns error), TestExtractorDim SKIP (model not present) or PASS (if model downloaded). All existing matcher/store tests still pass.

- [ ] **Step 5: Commit**

```bash
git add internal/speaker/extractor.go internal/speaker/extractor_test.go
git commit -m "feat: CAM++ speaker embedding extractor via sherpa-onnx"
```

---

## Task 4: Emotion Classifier

**Files:**
- Create: `internal/emotion/classifier.go`
- Create: `internal/emotion/classifier_test.go`

- [ ] **Step 1: Write the failing test**

Create `internal/emotion/classifier_test.go`:

```go
package emotion

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewClassifierRequiresModel(t *testing.T) {
	_, err := NewClassifier("/nonexistent/model", 1)
	if err == nil {
		t.Error("expected error for non-existent model path")
	}
}

func TestClassifyWithModel(t *testing.T) {
	modelDir := findSenseVoiceModel()
	if modelDir == "" {
		t.Skip("SenseVoice model not available, skipping")
	}

	cls, err := NewClassifier(modelDir, 1)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer cls.Close()

	// 1 second of silence at 16kHz
	samples := make([]float32, 16000)
	result, event, err := cls.Classify(samples, 16000)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	// Should return some emotion label (likely Neutral for silence)
	if result.Label == "" {
		t.Error("expected non-empty emotion label")
	}
	// Event should be a known value
	t.Logf("Emotion: %+v, Event: %s", result, event)
}

func findSenseVoiceModel() string {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".meeting-emo-transcriber", "models", "sensevoice-small-int8")
	if _, err := os.Stat(filepath.Join(dir, "model.int8.onnx")); err == nil {
		return dir
	}
	return ""
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
go test ./internal/emotion/ -v -run TestNewClassifierRequiresModel
```

Expected: FAIL — package does not exist.

- [ ] **Step 3: Implement classifier**

Create `internal/emotion/classifier.go`:

```go
package emotion

import (
	"fmt"
	"path/filepath"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos/sherpa_onnx"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// Classifier wraps sherpa-onnx OfflineRecognizer with SenseVoice model
// for emotion classification and audio event detection.
type Classifier struct {
	inner *sherpa.OfflineRecognizer
}

// NewClassifier creates an emotion classifier from a SenseVoice model directory.
// modelDir should contain model.int8.onnx and tokens.txt.
func NewClassifier(modelDir string, threads int) (*Classifier, error) {
	modelPath := filepath.Join(modelDir, "model.int8.onnx")
	tokensPath := filepath.Join(modelDir, "tokens.txt")

	config := &sherpa.OfflineRecognizerConfig{}
	config.ModelConfig.SenseVoice.Model = modelPath
	config.ModelConfig.SenseVoice.Language = ""
	config.ModelConfig.SenseVoice.UseInverseTextNormalization = 0
	config.ModelConfig.Tokens = tokensPath
	config.ModelConfig.NumThreads = threads
	config.ModelConfig.Debug = 0
	config.ModelConfig.Provider = "cpu"

	inner := sherpa.NewOfflineRecognizer(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create emotion classifier from %s", modelDir)
	}

	return &Classifier{inner: inner}, nil
}

// Classify performs emotion classification on audio samples.
// Returns the EmotionResult (3-layer mapping), audio event string, and error.
func (c *Classifier) Classify(samples []float32, sampleRate int) (types.EmotionResult, string, error) {
	stream := c.inner.CreateStream()
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)
	c.inner.Decode(stream)
	result := c.inner.GetResult(stream)

	emotionRaw := result.Emotion
	audioEvent := result.Event
	if audioEvent == "" {
		audioEvent = "Speech"
	}

	info := types.LookupEmotion(emotionRaw, types.SenseVoiceEmotionMap)

	return types.EmotionResult{
		Raw:        info.Raw,
		Label:      info.Label,
		Display:    info.Display,
		Confidence: 0, // SenseVoice via sherpa-onnx doesn't expose confidence scores directly
	}, audioEvent, nil
}

// Close releases the underlying sherpa-onnx resources.
func (c *Classifier) Close() {
	if c.inner != nil {
		sherpa.DeleteOfflineRecognizer(c.inner)
		c.inner = nil
	}
}
```

- [ ] **Step 4: Run tests**

```bash
go test ./internal/emotion/ -v
```

Expected: TestNewClassifierRequiresModel PASS, TestClassifyWithModel SKIP or PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/emotion/
git commit -m "feat: SenseVoice emotion classifier via sherpa-onnx"
```

---

## Task 5: Enroll Command — Real Embedding Computation

**Files:**
- Modify: `cmd/commands/enroll.go`

- [ ] **Step 1: Implement the full enroll command**

Replace `cmd/commands/enroll.go` with:

```go
package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newEnrollCmd() *cobra.Command {
	var force bool
	cmd := &cobra.Command{
		Use:   "enroll",
		Short: "Scan speakers/ directory and compute speaker embeddings",
		RunE: func(cmd *cobra.Command, args []string) error {
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			names, err := store.List()
			if err != nil {
				return err
			}
			if len(names) == 0 {
				fmt.Printf("No speaker directories found in %s/\n", speakersDir)
				return nil
			}

			// Extract embedded binaries (for ffmpeg)
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// Ensure speaker model
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}

			// Load config for threads
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}

			// Initialize extractor
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("create extractor: %w", err)
			}
			defer extractor.Close()

			fmt.Printf("Scanning %s/...\n", speakersDir)

			var created, updated, unchanged int
			for _, name := range names {
				files, err := store.ListAudioFiles(name)
				if err != nil {
					return err
				}
				if len(files) == 0 {
					fmt.Printf("  %s: no audio files, skipping\n", name)
					continue
				}

				needsUpdate := force
				if !force {
					needsUpdate, err = store.NeedsUpdate(name)
					if err != nil {
						return err
					}
				}

				if !needsUpdate {
					fmt.Printf("  %s: %d samples → unchanged (cached)\n", name, len(files))
					unchanged++
					continue
				}

				// Compute embeddings
				var embeddings []types.SampleEmbedding
				tmpDir, err := os.MkdirTemp("", "met-enroll-*")
				if err != nil {
					return fmt.Errorf("create temp dir: %w", err)
				}

				for _, file := range files {
					tempWav := filepath.Join(tmpDir, "enroll.wav")
					if err := audio.ConvertToWAV(bins.FFmpeg, file, tempWav); err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("convert %s: %w", file, err)
					}

					samples, sampleRate, err := audio.ReadWAV(tempWav)
					if err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("read %s: %w", file, err)
					}

					emb, err := extractor.Extract(samples, sampleRate)
					if err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("extract embedding from %s: %w", file, err)
					}

					hash, err := speaker.FileHash(file)
					if err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("hash %s: %w", file, err)
					}

					embeddings = append(embeddings, types.SampleEmbedding{
						File:      filepath.Base(file),
						Hash:      hash,
						Embedding: emb,
					})
				}
				os.RemoveAll(tmpDir)

				// Determine status
				status := "created"
				existing, _ := store.LoadProfile(name)
				if existing != nil {
					status = "updated"
				}

				now := time.Now().Format(time.RFC3339)
				profile := types.SpeakerProfile{
					Name:       name,
					Embeddings: embeddings,
					Dim:        extractor.Dim(),
					Model:      "campplus_sv_zh-cn",
					CreatedAt:  now,
					UpdatedAt:  now,
				}
				if existing != nil && existing.CreatedAt != "" {
					profile.CreatedAt = existing.CreatedAt
				}

				if err := store.SaveProfile(profile); err != nil {
					return fmt.Errorf("save profile %s: %w", name, err)
				}

				fmt.Printf("  %s: %d samples → embedding computed ✓ (%s)\n", name, len(files), status)
				if status == "created" {
					created++
				} else {
					updated++
				}
			}

			fmt.Printf("\n%d created, %d updated, %d unchanged.\n", created, updated, unchanged)
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "force recompute all embeddings")
	return cmd
}
```

- [ ] **Step 2: Add FileHash and LoadProfile to speaker store**

The enroll command uses `speaker.FileHash(file)` and `store.LoadProfile(name)` which may not be exported yet. Check `internal/speaker/store.go` — the `fileHash` function exists but is unexported. Export it:

Edit `internal/speaker/store.go`:
- Rename `fileHash` to `FileHash` (capitalize F)
- Add `LoadProfile(name string) (*types.SpeakerProfile, error)` method if not present

```go
// FileHash returns the SHA-256 hash of a file in "sha256:<hex>" format.
func FileHash(path string) (string, error) {
	// ... existing implementation with capitalized name
}

// LoadProfile loads a single speaker's profile from .profile.json.
// Returns nil, nil if the profile file doesn't exist.
func (s *Store) LoadProfile(name string) (*types.SpeakerProfile, error) {
	profilePath := filepath.Join(s.root, name, ".profile.json")
	data, err := os.ReadFile(profilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var profile types.SpeakerProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}
```

- [ ] **Step 3: Verify compilation**

```bash
go build ./cmd/main.go
```

Expected: BUILD SUCCESS.

- [ ] **Step 4: Run all tests**

```bash
go test ./... -v
```

Expected: All pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add cmd/commands/enroll.go internal/speaker/store.go
git commit -m "feat: enroll command with real CAM++ embedding computation"
```

---

## Task 6: Transcribe Command — Speaker + Emotion Integration

**Files:**
- Modify: `cmd/commands/transcribe.go`

- [ ] **Step 1: Update transcribe to add speaker/emotion inference**

Edit `cmd/commands/transcribe.go`. Replace the segment-building section (lines 88-101, the "Phase 2: no speaker/emotion" block) with the full pipeline:

After step 7 (ASR), add model loading and inference:

```go
			// 8. Ensure speaker and emotion models
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}
			emotionModelDir, err := models.EnsureModel("sensevoice-small-int8")
			if err != nil {
				return fmt.Errorf("ensure emotion model: %w", err)
			}

			// 9. Initialize speaker extractor and emotion classifier
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("create speaker extractor: %w", err)
			}
			defer extractor.Close()

			classifier, err := emotion.NewClassifier(emotionModelDir, cfg.Threads)
			if err != nil {
				return fmt.Errorf("create emotion classifier: %w", err)
			}
			defer classifier.Close()

			// 10. Load speaker profiles for matching
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})

			// 11. Read full WAV for segment extraction
			wavSamples, wavSampleRate, err := audio.ReadWAV(tempWavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}

			// 12. Process each ASR segment
			segments := make([]types.TranscriptSegment, 0, len(results))
			for _, r := range results {
				segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

				// Speaker identification
				speakerName := "Unknown"
				var speakerConf float32
				if len(segAudio) > 0 && len(profiles) > 0 {
					emb, embErr := extractor.Extract(segAudio, wavSampleRate)
					if embErr == nil {
						matchResult := matcher.Match(emb, profiles, float32(cfg.Threshold))
						if matchResult.Name != "" {
							speakerName = matchResult.Name
						}
						speakerConf = matchResult.Similarity
					}
				}

				// Emotion classification
				emotionInfo := types.EmotionInfo{Label: "Neutral", Display: ""}
				audioEvent := "Speech"
				var emotionConf float32
				if len(segAudio) > 0 {
					emotionResult, event, classErr := classifier.Classify(segAudio, wavSampleRate)
					if classErr == nil {
						emotionInfo = types.EmotionInfo{
							Raw:     emotionResult.Raw,
							Label:   emotionResult.Label,
							Display: emotionResult.Display,
						}
						audioEvent = event
						emotionConf = emotionResult.Confidence
					}
				}

				segments = append(segments, types.TranscriptSegment{
					Start:      r.Start,
					End:        r.End,
					Speaker:    speakerName,
					Emotion:    emotionInfo,
					AudioEvent: audioEvent,
					Language:   r.Language,
					Text:       r.Text,
					Confidence: types.Confidence{Speaker: speakerConf, Emotion: emotionConf},
				})
			}
```

Add new imports:
```go
"github.com/kouko/meeting-emo-transcriber/internal/emotion"
"github.com/kouko/meeting-emo-transcriber/internal/speaker"
```

Update Metadata to count speakers:
```go
			// Count speakers
			speakerSet := make(map[string]bool)
			identified := 0
			for _, seg := range segments {
				speakerSet[seg.Speaker] = true
				if seg.Speaker != "Unknown" {
					identified++
				}
			}

			transcript := types.TranscriptResult{
				Metadata: types.Metadata{
					File:               filepath.Base(inputPath),
					Duration:           time.Duration(duration * float64(time.Second)).String(),
					SpeakersDetected:   len(speakerSet),
					SpeakersIdentified: identified,
					Date:               time.Now().Format(time.RFC3339),
				},
				Segments: segments,
			}
```

- [ ] **Step 2: Verify compilation**

```bash
go build ./cmd/main.go
```

Expected: BUILD SUCCESS.

- [ ] **Step 3: Run all tests**

```bash
go test ./...
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add cmd/commands/transcribe.go
git commit -m "feat: transcribe with real speaker identification and emotion classification"
```

---

## Task 7: Build + E2E Verification

- [ ] **Step 1: Run all tests**

```bash
go test ./... -v -count=1
```

Expected: All pass.

- [ ] **Step 2: Build binary**

```bash
go build -o /tmp/met-phase3 ./cmd/main.go
```

Expected: BUILD SUCCESS.

- [ ] **Step 3: Verify CLI help**

```bash
/tmp/met-phase3 transcribe --help
/tmp/met-phase3 enroll --help
```

Expected: Shows all flags.

- [ ] **Step 4: Cleanup**

```bash
rm /tmp/met-phase3
```

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git diff --cached --stat
# Only commit if there are changes
git commit -m "chore: Phase 3 E2E verification fixes"
```

---

## Self-Review Checklist

- **Spec §2 (sherpa-onnx integration):** ✓ Task 1
- **Spec §3 (Speaker Extractor):** ✓ Task 3
- **Spec §4 (Emotion Classifier):** ✓ Task 4
- **Spec §5 (Model Registry):** ✓ Task 2
- **Spec §6 (CLI Integration):** ✓ Task 5 (enroll) + Task 6 (transcribe)
- **Spec §7 (File List):** ✓ All files covered
- **Spec §8 (Verification):** ✓ Task 7
- **Embedded adjustment:** ✓ Task 1 removes ONNXRuntime from BinPaths, extract, embed_prod, tests
- **No placeholders:** All code blocks complete
- **Type consistency:** `Extractor`, `Classifier`, `BinPaths`, `EmotionResult` consistent across tasks
