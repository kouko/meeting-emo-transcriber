# Phase 1: Foundation — Go 骨架、型別、Config、Speaker Store、Output Formatters、CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project foundation — all modules that don't depend on ONNX models or external binaries. After this phase, the CLI compiles and runs, config loads, speakers/ folder scanning works, output formatting works, and all pure-Go logic is tested.

**Architecture:** Go module with cobra CLI, viper config, folder-driven speaker store with SHA-256 hash caching, cosine similarity matching with strategy pattern, and three output formatters (TXT/JSON/SRT).

**Tech Stack:** Go 1.22+, spf13/cobra, spf13/viper, go-audio/wav, gonum

---

### Task 1: Go Module Init + Dependencies

**Files:**
- Create: `go.mod`
- Create: `go.sum`

- [ ] **Step 1: Initialize Go module**

```bash
cd /Users/kouko/GitHub/meeting-emo-transcriber
go mod init github.com/kouko/meeting-emo-transcriber
```

Expected: `go.mod` created with module path.

- [ ] **Step 2: Add dependencies**

```bash
go get github.com/spf13/cobra@latest
go get github.com/spf13/viper@latest
go get github.com/go-audio/wav@latest
go get github.com/go-audio/audio@latest
go get gonum.org/v1/gonum@latest
```

- [ ] **Step 3: Verify go.mod**

```bash
cat go.mod
```

Expected: Module path and all 5 dependencies listed.

- [ ] **Step 4: Commit**

```bash
git add go.mod go.sum
git commit -m "init: go module with core dependencies"
```

---

### Task 2: Core Types

**Files:**
- Create: `internal/types/types.go`
- Create: `internal/types/emotion.go`

- [ ] **Step 1: Create types package**

```go
// internal/types/types.go
package types

// AudioSegment represents an audio clip extracted from the original recording
// based on ASR timestamp boundaries. Used for Speaker Embedding and Emotion classification.
type AudioSegment struct {
	Start float64   // start time in seconds
	End   float64   // end time in seconds
	Audio []float32 // 16kHz mono PCM samples
}

// TranscriptSegment is a single segment in the final output.
type TranscriptSegment struct {
	Start      float64      `json:"start"`
	End        float64      `json:"end"`
	Speaker    string       `json:"speaker"`
	Emotion    EmotionInfo  `json:"emotion"`
	AudioEvent string       `json:"audio_event"`
	Language   string       `json:"language"`
	Text       string       `json:"text"`
	Confidence Confidence   `json:"confidence"`
}

// Confidence holds speaker and emotion confidence scores.
type Confidence struct {
	Speaker float32 `json:"speaker"`
	Emotion float32 `json:"emotion"`
}

// TranscriptResult is the complete transcription output.
type TranscriptResult struct {
	Metadata Metadata           `json:"metadata"`
	Segments []TranscriptSegment `json:"segments"`
}

// Metadata holds information about the transcription run.
type Metadata struct {
	File               string `json:"file"`
	Duration           string `json:"duration"`
	SpeakersDetected   int    `json:"speakers_detected"`
	SpeakersIdentified int    `json:"speakers_identified"`
	Date               string `json:"date"`
}

// SpeakerProfile represents a registered speaker's voiceprint (.profile.json).
type SpeakerProfile struct {
	Name       string            `json:"name"`
	Embeddings []SampleEmbedding `json:"embeddings"`
	Dim        int               `json:"dim"`
	Model      string            `json:"model"`
	CreatedAt  string            `json:"created_at"`
	UpdatedAt  string            `json:"updated_at"`
}

// SampleEmbedding holds one audio sample's embedding and file record.
type SampleEmbedding struct {
	File      string    `json:"file"`
	Hash      string    `json:"hash"` // "sha256:<hex>"
	Embedding []float32 `json:"embedding"`
}

// MatchResult holds the result of a speaker matching operation.
type MatchResult struct {
	Name       string
	Similarity float32
}

// EnrollResult holds the result of enrolling a single speaker.
type EnrollResult struct {
	Name    string
	Samples int
	Status  string // "created" | "updated" | "unchanged"
}

// ASRResult holds one segment from whisper-cli output.
type ASRResult struct {
	Start    float64
	End      float64
	Text     string
	Language string
}
```

- [ ] **Step 2: Create emotion types**

```go
// internal/types/emotion.go
package types

// EmotionInfo is the three-layer emotion representation for JSON output.
type EmotionInfo struct {
	Raw     string `json:"raw"`
	Label   string `json:"label"`
	Display string `json:"display"`
}

// EmotionResult is the internal emotion classification result.
type EmotionResult struct {
	Raw        string
	Label      string
	Display    string
	Confidence float32
}

// EmotionMapping maps raw model output → (label, display).
type EmotionMapping struct {
	Label   string
	Display string // CC Manner Caption adverb; empty for Neutral/Unknown/Other
}

// SenseVoiceEmotionMap maps SenseVoice raw tags to label+display.
var SenseVoiceEmotionMap = map[string]EmotionMapping{
	"HAPPY":     {Label: "Happy", Display: "happily"},
	"SAD":       {Label: "Sad", Display: "sadly"},
	"ANGRY":     {Label: "Angry", Display: "angrily"},
	"NEUTRAL":   {Label: "Neutral", Display: ""},
	"FEARFUL":   {Label: "Fearful", Display: "fearfully"},
	"DISGUSTED": {Label: "Disgusted", Display: "with disgust"},
	"SURPRISED": {Label: "Surprised", Display: "with surprise"},
	"unk":       {Label: "Unknown", Display: ""},
}

// Emotion2vecEmotionMap maps Emotion2vec+ raw labels to label+display.
var Emotion2vecEmotionMap = map[string]EmotionMapping{
	"angry":     {Label: "Angry", Display: "angrily"},
	"disgusted": {Label: "Disgusted", Display: "with disgust"},
	"fearful":   {Label: "Fearful", Display: "fearfully"},
	"happy":     {Label: "Happy", Display: "happily"},
	"neutral":   {Label: "Neutral", Display: ""},
	"other":     {Label: "Other", Display: ""},
	"sad":       {Label: "Sad", Display: "sadly"},
	"surprised": {Label: "Surprised", Display: "with surprise"},
	"unknown":   {Label: "Unknown", Display: ""},
}

// AudioEventDisplayMap maps SenseVoice audio event tags to CC display strings.
var AudioEventDisplayMap = map[string]string{
	"Speech":   "",
	"BGM":      "[background music]",
	"Applause": "[applause]",
	"Laughter": "[laughter]",
	"Cry":      "[crying]",
	"Sneeze":   "[sneeze]",
	"Breath":   "[breathing]",
	"Cough":    "[cough]",
}

// LookupEmotion resolves a raw emotion string using the given mapping.
// Returns EmotionInfo with all three layers populated.
func LookupEmotion(raw string, mapping map[string]EmotionMapping) EmotionInfo {
	if m, ok := mapping[raw]; ok {
		return EmotionInfo{Raw: raw, Label: m.Label, Display: m.Display}
	}
	return EmotionInfo{Raw: raw, Label: "Unknown", Display: ""}
}
```

- [ ] **Step 3: Verify it compiles**

```bash
go build ./internal/types/...
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add internal/types/
git commit -m "feat: add core types and emotion mappings"
```

---

### Task 3: Emotion Lookup Tests

**Files:**
- Create: `internal/types/emotion_test.go`

- [ ] **Step 1: Write tests**

```go
// internal/types/emotion_test.go
package types

import "testing"

func TestLookupEmotion_SenseVoice(t *testing.T) {
	tests := []struct {
		raw     string
		label   string
		display string
	}{
		{"HAPPY", "Happy", "happily"},
		{"SAD", "Sad", "sadly"},
		{"ANGRY", "Angry", "angrily"},
		{"NEUTRAL", "Neutral", ""},
		{"unk", "Unknown", ""},
		{"NONEXISTENT", "Unknown", ""},
	}
	for _, tt := range tests {
		info := LookupEmotion(tt.raw, SenseVoiceEmotionMap)
		if info.Label != tt.label {
			t.Errorf("LookupEmotion(%q).Label = %q, want %q", tt.raw, info.Label, tt.label)
		}
		if info.Display != tt.display {
			t.Errorf("LookupEmotion(%q).Display = %q, want %q", tt.raw, info.Display, tt.display)
		}
		if info.Raw != tt.raw {
			t.Errorf("LookupEmotion(%q).Raw = %q, want %q", tt.raw, info.Raw, tt.raw)
		}
	}
}

func TestLookupEmotion_Emotion2vec(t *testing.T) {
	info := LookupEmotion("happy", Emotion2vecEmotionMap)
	if info.Label != "Happy" || info.Display != "happily" {
		t.Errorf("got %+v", info)
	}

	info = LookupEmotion("other", Emotion2vecEmotionMap)
	if info.Label != "Other" || info.Display != "" {
		t.Errorf("got %+v", info)
	}
}
```

- [ ] **Step 2: Run tests**

```bash
go test ./internal/types/ -v
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add internal/types/emotion_test.go
git commit -m "test: emotion lookup mapping tests"
```

---

### Task 4: Speaker Matcher — Cosine Similarity + Strategy Pattern

**Files:**
- Create: `internal/speaker/matcher.go`
- Create: `internal/speaker/matcher_test.go`

- [ ] **Step 1: Write failing tests**

```go
// internal/speaker/matcher_test.go
package speaker

import (
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	if sim < 0.999 {
		t.Errorf("identical vectors: got %f, want ~1.0", sim)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	sim := CosineSimilarity(a, b)
	if sim > 0.001 {
		t.Errorf("orthogonal vectors: got %f, want ~0.0", sim)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{-1, 0, 0}
	sim := CosineSimilarity(a, b)
	if sim > -0.999 {
		t.Errorf("opposite vectors: got %f, want ~-1.0", sim)
	}
}

func TestMaxSimilarityStrategy_BestMatch(t *testing.T) {
	strategy := &MaxSimilarityStrategy{}
	profile := types.SpeakerProfile{
		Name: "TestUser",
		Embeddings: []types.SampleEmbedding{
			{Embedding: []float32{1, 0, 0}},
			{Embedding: []float32{0, 1, 0}},
			{Embedding: []float32{0.7, 0.7, 0}},
		},
	}
	// query close to third sample
	query := []float32{0.71, 0.71, 0}
	score := strategy.Score(query, profile)
	if score < 0.99 {
		t.Errorf("expected high score for close match, got %f", score)
	}
}

func TestMatcherMatch_AboveThreshold(t *testing.T) {
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	profiles := []types.SpeakerProfile{
		{
			Name: "Alice",
			Embeddings: []types.SampleEmbedding{
				{Embedding: []float32{1, 0, 0}},
			},
		},
		{
			Name: "Bob",
			Embeddings: []types.SampleEmbedding{
				{Embedding: []float32{0, 1, 0}},
			},
		},
	}
	result := matcher.Match([]float32{0.99, 0.1, 0}, profiles, 0.6)
	if result.Name != "Alice" {
		t.Errorf("expected Alice, got %q", result.Name)
	}
}

func TestMatcherMatch_BelowThreshold(t *testing.T) {
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	profiles := []types.SpeakerProfile{
		{
			Name: "Alice",
			Embeddings: []types.SampleEmbedding{
				{Embedding: []float32{1, 0, 0}},
			},
		},
	}
	result := matcher.Match([]float32{0, 1, 0}, profiles, 0.6)
	if result.Name != "" {
		t.Errorf("expected empty name for no match, got %q", result.Name)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/speaker/ -v
```

Expected: FAIL — package doesn't exist yet.

- [ ] **Step 3: Implement matcher**

```go
// internal/speaker/matcher.go
package speaker

import (
	"math"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// MatchStrategy defines how to score a segment embedding against a speaker profile.
type MatchStrategy interface {
	Score(segmentEmb []float32, profile types.SpeakerProfile) float32
}

// MaxSimilarityStrategy scores by comparing against every sample and taking the max.
type MaxSimilarityStrategy struct{}

func (s *MaxSimilarityStrategy) Score(segmentEmb []float32, profile types.SpeakerProfile) float32 {
	var maxSim float32 = -1
	for _, sample := range profile.Embeddings {
		sim := CosineSimilarity(segmentEmb, sample.Embedding)
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
}

// Matcher matches segment embeddings against registered speaker profiles.
type Matcher struct {
	strategy MatchStrategy
}

// NewMatcher creates a Matcher with the given strategy.
func NewMatcher(strategy MatchStrategy) *Matcher {
	return &Matcher{strategy: strategy}
}

// Match finds the best matching speaker profile above the threshold.
// Returns MatchResult with empty Name if no profile exceeds threshold.
func (m *Matcher) Match(embedding []float32, profiles []types.SpeakerProfile, threshold float32) types.MatchResult {
	var bestName string
	var bestSim float32 = -1

	for _, profile := range profiles {
		sim := m.strategy.Score(embedding, profile)
		if sim > bestSim {
			bestSim = sim
			bestName = profile.Name
		}
	}

	if bestSim < threshold {
		return types.MatchResult{Name: "", Similarity: bestSim}
	}
	return types.MatchResult{Name: bestName, Similarity: bestSim}
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./internal/speaker/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/speaker/matcher.go internal/speaker/matcher_test.go
git commit -m "feat: speaker matcher with cosine similarity and MaxSimilarity strategy"
```

---

### Task 5: Speaker Store — Folder Scanning + Hash Caching

**Files:**
- Create: `internal/speaker/store.go`
- Create: `internal/speaker/store_test.go`

- [ ] **Step 1: Write failing tests**

```go
// internal/speaker/store_test.go
package speaker

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStore_ListEmpty(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 0 {
		t.Errorf("expected empty list, got %v", names)
	}
}

func TestStore_ListWithSpeakers(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "Alice"), 0755)
	os.MkdirAll(filepath.Join(dir, "Bob"), 0755)
	// config.yaml should not be listed as a speaker
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(""), 0644)

	store := NewStore(dir, supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Errorf("expected 2 speakers, got %d: %v", len(names), names)
	}
}

func TestStore_LoadProfiles_NoProfile(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "Alice"), 0755)

	store := NewStore(dir, supportedExtensions())
	profiles, err := store.LoadProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if len(profiles) != 0 {
		t.Errorf("expected 0 profiles (no .profile.json), got %d", len(profiles))
	}
}

func TestStore_FileHash_Deterministic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wav")
	os.WriteFile(path, []byte("fake audio data"), 0644)

	h1, err := fileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := fileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Errorf("hash not deterministic: %q != %q", h1, h2)
	}
	if len(h1) < 10 {
		t.Errorf("hash too short: %q", h1)
	}
}

func supportedExtensions() []string {
	return []string{".wav", ".mp3", ".m4a", ".flac", ".ogg"}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/speaker/ -v -run TestStore
```

Expected: FAIL — `NewStore` not defined.

- [ ] **Step 3: Implement store**

```go
// internal/speaker/store.go
package speaker

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

const profileFileName = ".profile.json"

// Store manages speaker profiles in a folder-driven structure.
type Store struct {
	dir        string
	extensions []string // supported audio extensions (e.g., ".wav", ".mp3")
}

// NewStore creates a Store that scans the given directory.
func NewStore(dir string, extensions []string) *Store {
	return &Store{dir: dir, extensions: extensions}
}

// List returns the names of all speaker subdirectories.
func (s *Store) List() ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading speakers dir: %w", err)
	}

	var names []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			names = append(names, e.Name())
		}
	}
	return names, nil
}

// LoadProfiles loads all speaker profiles that have a valid .profile.json.
func (s *Store) LoadProfiles() ([]types.SpeakerProfile, error) {
	names, err := s.List()
	if err != nil {
		return nil, err
	}

	var profiles []types.SpeakerProfile
	for _, name := range names {
		profilePath := filepath.Join(s.dir, name, profileFileName)
		data, err := os.ReadFile(profilePath)
		if err != nil {
			continue // no profile yet — skip
		}
		var profile types.SpeakerProfile
		if err := json.Unmarshal(data, &profile); err != nil {
			continue // corrupted — skip
		}
		profiles = append(profiles, profile)
	}
	return profiles, nil
}

// SaveProfile writes a speaker profile to .profile.json.
func (s *Store) SaveProfile(profile types.SpeakerProfile) error {
	dir := filepath.Join(s.dir, profile.Name)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating speaker dir: %w", err)
	}

	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling profile: %w", err)
	}
	return os.WriteFile(filepath.Join(dir, profileFileName), data, 0644)
}

// ListAudioFiles returns all audio files in a speaker's directory.
func (s *Store) ListAudioFiles(speakerName string) ([]string, error) {
	dir := filepath.Join(s.dir, speakerName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading speaker dir %q: %w", speakerName, err)
	}

	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		for _, supported := range s.extensions {
			if ext == supported {
				files = append(files, e.Name())
				break
			}
		}
	}
	return files, nil
}

// NeedsUpdate checks if a speaker's .profile.json is stale by comparing
// audio file hashes against the cached hashes in the profile.
func (s *Store) NeedsUpdate(speakerName string) (bool, error) {
	profilePath := filepath.Join(s.dir, speakerName, profileFileName)
	data, err := os.ReadFile(profilePath)
	if err != nil {
		return true, nil // no profile = needs creation
	}

	var profile types.SpeakerProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return true, nil // corrupted = needs recreation
	}

	audioFiles, err := s.ListAudioFiles(speakerName)
	if err != nil {
		return false, err
	}

	// Build hash set from cached profile
	cachedHashes := make(map[string]string) // file → hash
	for _, sample := range profile.Embeddings {
		cachedHashes[sample.File] = sample.Hash
	}

	// Compare: different count = needs update
	if len(audioFiles) != len(cachedHashes) {
		return true, nil
	}

	// Compare each file's hash
	for _, file := range audioFiles {
		cachedHash, exists := cachedHashes[file]
		if !exists {
			return true, nil // new file
		}
		currentHash, err := fileHash(filepath.Join(s.dir, speakerName, file))
		if err != nil {
			return false, err
		}
		if currentHash != cachedHash {
			return true, nil // file changed
		}
	}

	return false, nil
}

// fileHash computes SHA-256 hash of a file, returned as "sha256:<hex>".
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
	return fmt.Sprintf("sha256:%x", h.Sum(nil)), nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./internal/speaker/ -v -run TestStore
```

Expected: All store tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/speaker/store.go internal/speaker/store_test.go
git commit -m "feat: speaker store with folder scanning and hash caching"
```

---

### Task 6: Config Loading

**Files:**
- Create: `internal/config/config.go`
- Create: `internal/config/config_test.go`

- [ ] **Step 1: Write failing tests**

```go
// internal/config/config_test.go
package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()
	if cfg.Language != "auto" {
		t.Errorf("Language = %q, want auto", cfg.Language)
	}
	if cfg.Threshold != 0.6 {
		t.Errorf("Threshold = %f, want 0.6", cfg.Threshold)
	}
	if cfg.Format != "txt" {
		t.Errorf("Format = %q, want txt", cfg.Format)
	}
	if cfg.Strategy != "max_similarity" {
		t.Errorf("Strategy = %q, want max_similarity", cfg.Strategy)
	}
	if !cfg.Discover {
		t.Error("Discover should default to true")
	}
}

func TestLoad_FromSpeakersDir(t *testing.T) {
	dir := t.TempDir()
	yaml := `language: "zh-TW"
threshold: 0.8
format: json
`
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(yaml), 0644)

	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Language != "zh-TW" {
		t.Errorf("Language = %q, want zh-TW", cfg.Language)
	}
	if cfg.Threshold != 0.8 {
		t.Errorf("Threshold = %f, want 0.8", cfg.Threshold)
	}
	if cfg.Format != "json" {
		t.Errorf("Format = %q, want json", cfg.Format)
	}
	// Discover should still be true (default, not overridden)
	if !cfg.Discover {
		t.Error("Discover should default to true when not in config")
	}
}

func TestLoad_NoConfigFile(t *testing.T) {
	dir := t.TempDir()
	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	// Should return defaults
	if cfg.Language != "auto" {
		t.Errorf("Language = %q, want auto", cfg.Language)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/config/ -v
```

Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Implement config**

```go
// internal/config/config.go
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/viper"
)

// Config holds all application settings.
type Config struct {
	Language  string  `mapstructure:"language"`
	Threshold float32 `mapstructure:"threshold"`
	Format    string  `mapstructure:"format"`
	Strategy  string  `mapstructure:"strategy"`
	Discover  bool    `mapstructure:"discover"`
	LogLevel  string  `mapstructure:"log_level"`
	Threads   int     `mapstructure:"threads"`

	Models struct {
		Whisper string `mapstructure:"whisper"`
		Speaker string `mapstructure:"speaker"`
		Emotion string `mapstructure:"emotion"`
	} `mapstructure:"models"`
}

// Defaults returns a Config with all default values.
func Defaults() Config {
	homeDir, _ := os.UserHomeDir()
	base := filepath.Join(homeDir, ".meeting-emo-transcriber", "models")

	return Config{
		Language:  "auto",
		Threshold: 0.6,
		Format:    "txt",
		Strategy:  "max_similarity",
		Discover:  true,
		LogLevel:  "info",
		Threads:   runtime.NumCPU(),
		Models: struct {
			Whisper string `mapstructure:"whisper"`
			Speaker string `mapstructure:"speaker"`
			Emotion string `mapstructure:"emotion"`
		}{
			Speaker: filepath.Join(base, "campplus_sv_zh-cn.onnx"),
			Emotion: filepath.Join(base, "sensevoice-small-int8.onnx"),
			// Whisper intentionally empty — resolved by language auto-selection
		},
	}
}

// Load reads config from an explicit path or from <speakersDir>/config.yaml.
// Empty configPath means "use speakers dir config if it exists".
func Load(configPath string, speakersDir string) (Config, error) {
	cfg := Defaults()

	v := viper.New()
	v.SetConfigType("yaml")

	// Determine config file path
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else if speakersDir != "" {
		candidate := filepath.Join(speakersDir, "config.yaml")
		if _, err := os.Stat(candidate); err == nil {
			v.SetConfigFile(candidate)
		} else {
			return cfg, nil // no config file — use defaults
		}
	} else {
		return cfg, nil
	}

	if err := v.ReadInConfig(); err != nil {
		return cfg, fmt.Errorf("reading config: %w", err)
	}

	if err := v.Unmarshal(&cfg); err != nil {
		return cfg, fmt.Errorf("parsing config: %w", err)
	}

	return cfg, nil
}

// ParseFormats splits a format string like "txt,json,srt" into a slice.
// "all" expands to all supported formats.
func ParseFormats(format string) []string {
	if format == "all" {
		return []string{"txt", "json", "srt"}
	}
	parts := strings.Split(format, ",")
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}

// SupportedAudioExtensions returns the list of audio file extensions for speaker scanning.
func SupportedAudioExtensions() []string {
	return []string{".wav", ".mp3", ".m4a", ".flac", ".ogg"}
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./internal/config/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/config/
git commit -m "feat: config loading with viper, defaults, and speakers dir support"
```

---

### Task 7: Output Formatters — TXT

**Files:**
- Create: `internal/output/txt.go`
- Create: `internal/output/txt_test.go`

- [ ] **Step 1: Write failing tests**

```go
// internal/output/txt_test.go
package output

import (
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestTXTFormatter_BasicOutput(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Hello everyone"},
			{Speaker: "Bob", Emotion: types.EmotionInfo{Display: ""}, Text: "Hi Alice"},
		},
	}

	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out, "Alice [happily]") {
		t.Error("expected 'Alice [happily]' in output")
	}
	if !strings.Contains(out, "Hello everyone") {
		t.Error("expected text in output")
	}
	// Bob has Neutral (empty display) — no emotion tag
	if strings.Contains(out, "Bob [") {
		t.Error("Bob should not have emotion tag (Neutral)")
	}
	if !strings.Contains(out, "Bob\n") {
		t.Error("expected 'Bob' on its own line")
	}
}

func TestTXTFormatter_MergeSameSpeaker(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "First line"},
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Second line"},
			{Speaker: "Bob", Emotion: types.EmotionInfo{Display: ""}, Text: "Bob speaks"},
		},
	}

	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	// Alice should appear only once as header
	if strings.Count(out, "Alice") != 1 {
		t.Errorf("Alice should appear once, got:\n%s", out)
	}
	if !strings.Contains(out, "First line\nSecond line") {
		t.Errorf("consecutive lines should be merged:\n%s", out)
	}
}

func TestTXTFormatter_EmotionChangeInline(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Good news"},
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "angrily"}, Text: "But this is bad"},
		},
	}

	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out, "[angrily]") {
		t.Errorf("expected inline emotion change:\n%s", out)
	}
}

func TestTXTFormatter_AudioEvent(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: ""}, AudioEvent: "Laughter", Text: ""},
		},
	}

	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out, "[laughter]") {
		t.Errorf("expected audio event in output:\n%s", out)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
go test ./internal/output/ -v
```

Expected: FAIL.

- [ ] **Step 3: Implement TXT formatter**

```go
// internal/output/txt.go
package output

import (
	"fmt"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// TXTFormatter outputs plain text transcript with CC Manner Caption conventions.
type TXTFormatter struct{}

func (f *TXTFormatter) Format(result types.TranscriptResult) (string, error) {
	var b strings.Builder

	var currentSpeaker string
	var currentEmotion string

	for _, seg := range result.Segments {
		// Handle non-speech audio events
		eventDisplay := types.AudioEventDisplayMap[seg.AudioEvent]
		if eventDisplay != "" && seg.Text == "" {
			// Standalone audio event (e.g., [laughter] with no speech)
			if currentSpeaker != "" {
				fmt.Fprintf(&b, "%s\n", eventDisplay)
			} else {
				fmt.Fprintf(&b, "%s\n", eventDisplay)
			}
			continue
		}

		speakerChanged := seg.Speaker != currentSpeaker

		if speakerChanged {
			// Add blank line between speakers (except at start)
			if currentSpeaker != "" {
				b.WriteString("\n")
			}
			currentSpeaker = seg.Speaker
			currentEmotion = ""

			// Speaker header line
			if seg.Emotion.Display != "" {
				fmt.Fprintf(&b, "%s [%s]\n", seg.Speaker, seg.Emotion.Display)
				currentEmotion = seg.Emotion.Display
			} else {
				fmt.Fprintf(&b, "%s\n", seg.Speaker)
			}
		} else if seg.Emotion.Display != currentEmotion && seg.Emotion.Display != "" {
			// Same speaker, emotion changed — inline tag
			fmt.Fprintf(&b, "[%s] ", seg.Emotion.Display)
			currentEmotion = seg.Emotion.Display
		}

		// Audio event prefix (for speech segments with events like laughter)
		if eventDisplay != "" {
			fmt.Fprintf(&b, "%s ", eventDisplay)
		}

		if seg.Text != "" {
			fmt.Fprintf(&b, "%s\n", seg.Text)
		}
	}

	return b.String(), nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
go test ./internal/output/ -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/output/txt.go internal/output/txt_test.go
git commit -m "feat: TXT output formatter with CC Manner Caption conventions"
```

---

### Task 8: Output Formatters — JSON

**Files:**
- Create: `internal/output/json.go`
- Create: `internal/output/json_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/output/json_test.go
package output

import (
	"encoding/json"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestJSONFormatter_ValidJSON(t *testing.T) {
	result := types.TranscriptResult{
		Metadata: types.Metadata{File: "test.wav", Duration: "00:01:00"},
		Segments: []types.TranscriptSegment{
			{
				Start:   0.0,
				End:     5.0,
				Speaker: "Alice",
				Emotion: types.EmotionInfo{Raw: "HAPPY", Label: "Happy", Display: "happily"},
				Text:    "Hello",
			},
		},
	}

	f := &JSONFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	// Verify it's valid JSON
	var parsed types.TranscriptResult
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, out)
	}

	if parsed.Segments[0].Emotion.Raw != "HAPPY" {
		t.Errorf("emotion.raw = %q, want HAPPY", parsed.Segments[0].Emotion.Raw)
	}
}
```

- [ ] **Step 2: Implement JSON formatter**

```go
// internal/output/json.go
package output

import (
	"encoding/json"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// JSONFormatter outputs the full TranscriptResult as indented JSON.
type JSONFormatter struct{}

func (f *JSONFormatter) Format(result types.TranscriptResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data) + "\n", nil
}
```

- [ ] **Step 3: Run tests**

```bash
go test ./internal/output/ -v -run TestJSON
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/output/json.go internal/output/json_test.go
git commit -m "feat: JSON output formatter"
```

---

### Task 9: Output Formatters — SRT

**Files:**
- Create: `internal/output/srt.go`
- Create: `internal/output/srt_test.go`

- [ ] **Step 1: Write failing test**

```go
// internal/output/srt_test.go
package output

import (
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestSRTFormatter_DCMPFormat(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Start: 0.0, End: 15.2, Speaker: "CEO_Wang", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Great results"},
			{Start: 15.2, End: 30.0, Speaker: "Manager_Lin", Emotion: types.EmotionInfo{Display: ""}, Text: "Here are the numbers"},
		},
	}

	f := &SRTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(out, "(CEO_Wang) [happily] Great results") {
		t.Errorf("expected DCMP format:\n%s", out)
	}
	// Neutral emotion — no tag
	if !strings.Contains(out, "(Manager_Lin) Here are the numbers") {
		t.Errorf("expected no emotion tag for Neutral:\n%s", out)
	}
	if !strings.Contains(out, "00:00:00,000 --> 00:00:15,200") {
		t.Errorf("expected SRT timestamp format:\n%s", out)
	}
}

func TestFormatSRTTimestamp(t *testing.T) {
	tests := []struct {
		seconds float64
		want    string
	}{
		{0.0, "00:00:00,000"},
		{15.2, "00:00:15,200"},
		{90.5, "00:01:30,500"},
		{3661.123, "01:01:01,123"},
	}
	for _, tt := range tests {
		got := formatSRTTimestamp(tt.seconds)
		if got != tt.want {
			t.Errorf("formatSRTTimestamp(%f) = %q, want %q", tt.seconds, got, tt.want)
		}
	}
}
```

- [ ] **Step 2: Implement SRT formatter**

```go
// internal/output/srt.go
package output

import (
	"fmt"
	"math"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// SRTFormatter outputs SRT subtitles with DCMP CC caption conventions.
type SRTFormatter struct{}

func (f *SRTFormatter) Format(result types.TranscriptResult) (string, error) {
	var b strings.Builder

	for i, seg := range result.Segments {
		// Sequence number
		fmt.Fprintf(&b, "%d\n", i+1)

		// Timestamps
		fmt.Fprintf(&b, "%s --> %s\n",
			formatSRTTimestamp(seg.Start),
			formatSRTTimestamp(seg.End))

		// Content line: (Speaker) [emotion] text
		var line strings.Builder
		fmt.Fprintf(&line, "(%s)", seg.Speaker)

		if seg.Emotion.Display != "" {
			fmt.Fprintf(&line, " [%s]", seg.Emotion.Display)
		}

		fmt.Fprintf(&line, " %s", seg.Text)
		fmt.Fprintf(&b, "%s\n\n", strings.TrimSpace(line.String()))
	}

	return b.String(), nil
}

// formatSRTTimestamp converts seconds to SRT timestamp format HH:MM:SS,mmm.
func formatSRTTimestamp(seconds float64) string {
	totalMs := int(math.Round(seconds * 1000))
	h := totalMs / 3600000
	totalMs %= 3600000
	m := totalMs / 60000
	totalMs %= 60000
	s := totalMs / 1000
	ms := totalMs % 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}
```

- [ ] **Step 3: Run tests**

```bash
go test ./internal/output/ -v -run TestSRT
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add internal/output/srt.go internal/output/srt_test.go
git commit -m "feat: SRT output formatter with DCMP CC conventions"
```

---

### Task 10: CLI Skeleton

**Files:**
- Create: `cmd/main.go`
- Create: `cmd/commands/root.go`
- Create: `cmd/commands/init_cmd.go`
- Create: `cmd/commands/transcribe.go`
- Create: `cmd/commands/enroll.go`
- Create: `cmd/commands/speakers.go`

- [ ] **Step 1: Create root command**

```go
// cmd/commands/root.go
package commands

import (
	"github.com/spf13/cobra"
)

var (
	speakersDir string
	configPath  string
	logLevel    string
)

func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "meeting-emo-transcriber",
		Short: "Meeting transcriber with speaker identification and emotion recognition",
	}

	root.PersistentFlags().StringVar(&speakersDir, "speakers", "./speakers", "speakers directory path")
	root.PersistentFlags().StringVar(&configPath, "config", "", "config file path (default: <speakers-dir>/config.yaml)")
	root.PersistentFlags().StringVar(&logLevel, "log-level", "info", "log level: debug|info|warn|error")

	root.AddCommand(newInitCmd())
	root.AddCommand(newTranscribeCmd())
	root.AddCommand(newEnrollCmd())
	root.AddCommand(newSpeakersCmd())

	return root
}
```

- [ ] **Step 2: Create init command**

```go
// cmd/commands/init_cmd.go
package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

func newInitCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Initialize working directory (creates speakers/ + output/ + config.yaml template)",
		RunE: func(cmd *cobra.Command, args []string) error {
			dirs := []string{"speakers", "output"}
			for _, d := range dirs {
				if err := os.MkdirAll(d, 0755); err != nil {
					return fmt.Errorf("creating %s: %w", d, err)
				}
				fmt.Printf("  Created %s/\n", d)
			}

			configPath := filepath.Join("speakers", "config.yaml")
			if _, err := os.Stat(configPath); os.IsNotExist(err) {
				template := `# speakers/config.yaml
# Uncomment and modify values as needed. All have sensible defaults.

# language: "auto"        # auto | zh-TW | zh | en | ja
# threshold: 0.6          # Speaker similarity threshold
# format: txt             # txt | json | srt | all
# discover: true          # Auto-discover unknown speakers
`
				if err := os.WriteFile(configPath, []byte(template), 0644); err != nil {
					return fmt.Errorf("writing config template: %w", err)
				}
				fmt.Printf("  Created %s\n", configPath)
			}

			fmt.Println("\nWorkspace initialized. Add speaker audio samples to speakers/<name>/ and run 'transcribe'.")
			return nil
		},
	}
}
```

- [ ] **Step 3: Create transcribe command stub**

```go
// cmd/commands/transcribe.go
package commands

import (
	"fmt"

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
			// TODO: implement in Phase 2+
			fmt.Printf("Transcribe: input=%s speakers=%s format=%s language=%s\n",
				inputPath, speakersDir, format, language)
			fmt.Println("(not yet implemented — Phase 2)")
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
```

- [ ] **Step 4: Create enroll command stub**

```go
// cmd/commands/enroll.go
package commands

import (
	"fmt"

	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/spf13/cobra"
)

func newEnrollCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:   "enroll",
		Short: "Scan speakers/ directory and register all speakers",
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

			fmt.Printf("Scanning %s/...\n", speakersDir)
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

				if needsUpdate {
					// TODO: compute embeddings in Phase 3
					fmt.Printf("  %s: %d samples → embedding computation pending (Phase 3)\n", name, len(files))
				} else {
					fmt.Printf("  %s: %d samples → unchanged (cached)\n", name, len(files))
				}
			}
			return nil
		},
	}

	cmd.Flags().BoolVar(&force, "force", false, "force recompute all embeddings")
	return cmd
}
```

- [ ] **Step 5: Create speakers command stub**

```go
// cmd/commands/speakers.go
package commands

import (
	"fmt"

	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/spf13/cobra"
)

func newSpeakersCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "speakers",
		Short: "Speaker management commands",
	}

	cmd.AddCommand(newSpeakersListCmd())
	cmd.AddCommand(newSpeakersVerifyCmd())
	return cmd
}

func newSpeakersListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all registered speakers",
		RunE: func(cmd *cobra.Command, args []string) error {
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			names, err := store.List()
			if err != nil {
				return err
			}
			if len(names) == 0 {
				fmt.Println("No speakers found.")
				return nil
			}
			for _, name := range names {
				files, _ := store.ListAudioFiles(name)
				needsUpdate, _ := store.NeedsUpdate(name)
				status := "enrolled"
				if needsUpdate {
					status = "needs enrollment"
				}
				fmt.Printf("  %s: %d samples (%s)\n", name, len(files), status)
			}
			return nil
		},
	}
}

func newSpeakersVerifyCmd() *cobra.Command {
	var name, audio string

	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify speaker recognition accuracy",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: implement in Phase 3
			fmt.Printf("Verify: name=%s audio=%s\n", name, audio)
			fmt.Println("(not yet implemented — Phase 3)")
			return nil
		},
	}

	cmd.Flags().StringVar(&name, "name", "", "speaker name (required)")
	cmd.Flags().StringVar(&audio, "audio", "", "test audio file path (required)")
	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("audio")
	return cmd
}
```

- [ ] **Step 6: Create main.go entry point**

```go
// cmd/main.go
package main

import (
	"fmt"
	"os"

	"github.com/kouko/meeting-emo-transcriber/cmd/commands"
)

func main() {
	if err := commands.NewRootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
```

- [ ] **Step 7: Build and test CLI**

```bash
go build -o meeting-emo-transcriber ./cmd/main.go
./meeting-emo-transcriber --help
./meeting-emo-transcriber init
./meeting-emo-transcriber speakers list
./meeting-emo-transcriber transcribe --input test.wav
```

Expected:
- `--help` shows all commands
- `init` creates `speakers/` + `output/` + `speakers/config.yaml`
- `speakers list` shows empty list or speakers found
- `transcribe` shows stub message

- [ ] **Step 8: Run all tests**

```bash
go test ./... -v
```

Expected: All tests PASS across all packages.

- [ ] **Step 9: Commit**

```bash
git add cmd/ internal/
git commit -m "feat: CLI skeleton with init, transcribe, enroll, speakers commands"
```

---

## Self-Review Checklist

1. **Spec coverage**: §4 types ✓, §5 matcher interface ✓, §5 store interface ✓, §7 CLI commands ✓, §8 config ✓, §9 TXT/JSON/SRT formatters ✓. Deferred to Phase 2+: §6 pipeline, embedded, audio, asr, emotion, speaker/extractor.

2. **Placeholder scan**: All `TODO` comments explicitly note which phase will implement them. No "add appropriate handling" placeholders.

3. **Type consistency**: `types.SpeakerProfile`, `types.MatchResult`, `types.EmotionInfo`, `types.TranscriptResult` used consistently across all tasks. `speaker.Store` and `speaker.Matcher` follow §5 interfaces.
