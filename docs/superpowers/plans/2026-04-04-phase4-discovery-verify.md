# Phase 4: Unknown Speaker Discovery + Speakers Verify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-discover and persist unknown speakers during transcription (speaker_1, speaker_2...) with cross-segment clustering, and implement the `speakers verify` command for testing speaker recognition accuracy.

**Architecture:** A `Discovery` struct in `internal/speaker/` manages unknown speaker tracking per transcription session. It wraps the existing `Matcher` for known-speaker lookup, adds cosine-similarity matching against session-local unknowns for clustering, and persists new speakers to `speakers/speaker_N/` with audio samples and `.profile.json`. The `--no-discover` flag disables persistence while keeping clustering. The `speakers verify` command reuses existing `Extractor` and `Matcher`.

**Tech Stack:** Go 1.25, sherpa-onnx-go-macos (CAM++ extractor), existing speaker/matcher/store packages

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `internal/speaker/discovery.go` | Unknown speaker tracking, clustering, and persistence |
| `internal/speaker/discovery_test.go` | Unit tests for discovery logic |

### Modified files

| File | Changes |
|------|---------|
| `internal/speaker/store.go` | Add `Root() string` method |
| `cmd/commands/transcribe.go` | Replace direct matcher call with Discovery |
| `cmd/commands/speakers.go` | Implement `speakers verify` command |

---

## Task 1: Store.Root() Method

**Files:**
- Modify: `internal/speaker/store.go`

- [ ] **Step 1: Write the failing test**

Add to `internal/speaker/store_test.go`:

```go
func TestStoreRoot(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	if store.Root() != dir {
		t.Errorf("Root() = %q, want %q", store.Root(), dir)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./internal/speaker/ -v -run TestStoreRoot`
Expected: FAIL — `store.Root undefined`

- [ ] **Step 3: Implement Root()**

Add to `internal/speaker/store.go`:

```go
// Root returns the root directory path of the speaker store.
func (s *Store) Root() string {
	return s.root
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./internal/speaker/ -v -run TestStoreRoot`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add internal/speaker/store.go internal/speaker/store_test.go
git commit -m "feat: add Root() method to speaker Store"
```

---

## Task 2: Discovery Module

**Files:**
- Create: `internal/speaker/discovery.go`
- Create: `internal/speaker/discovery_test.go`

- [ ] **Step 1: Write the failing tests**

Create `internal/speaker/discovery_test.go`:

```go
package speaker

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// mockExtractor is a fake Extractor for testing (no ONNX model needed).
type mockExtractor struct {
	dim int
}

func (m *mockExtractor) Dim() int { return m.dim }

func TestDiscoveryKnownSpeaker(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false)

	// Create a profile with a known embedding
	profiles := []types.SpeakerProfile{
		{
			Name: "Alice",
			Embeddings: []types.SampleEmbedding{
				{Embedding: makeEmbedding(512, 1.0)},
			},
		},
	}

	// An embedding identical to Alice's should match
	name, conf := d.IdentifySpeaker(
		makeEmbedding(512, 1.0), profiles, nil, 16000, 0.0,
	)
	if name != "Alice" {
		t.Errorf("expected Alice, got %q", name)
	}
	if conf < 0.9 {
		t.Errorf("expected high confidence, got %f", conf)
	}
}

func TestDiscoveryNewUnknown(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false)

	// Empty profiles → no known match → creates speaker_1
	name, _ := d.IdentifySpeaker(
		makeEmbedding(512, 1.0), nil, nil, 16000, 1.5,
	)
	if name != "speaker_1" {
		t.Errorf("expected speaker_1, got %q", name)
	}

	if len(d.Unknowns()) != 1 {
		t.Errorf("expected 1 unknown, got %d", len(d.Unknowns()))
	}
}

func TestDiscoveryCrossSegmentClustering(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false)

	emb := makeEmbedding(512, 1.0)

	// First call creates speaker_1
	name1, _ := d.IdentifySpeaker(emb, nil, nil, 16000, 0.0)

	// Second call with same embedding should cluster to speaker_1
	name2, _ := d.IdentifySpeaker(emb, nil, nil, 16000, 5.0)

	if name1 != name2 {
		t.Errorf("cross-segment clustering failed: %q != %q", name1, name2)
	}
	if name1 != "speaker_1" {
		t.Errorf("expected speaker_1, got %q", name1)
	}
}

func TestDiscoveryMultipleUnknowns(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false)

	// Two very different embeddings → two different unknowns
	emb1 := makeEmbedding(512, 1.0)
	emb2 := makeEmbedding(512, -1.0)

	name1, _ := d.IdentifySpeaker(emb1, nil, nil, 16000, 0.0)
	name2, _ := d.IdentifySpeaker(emb2, nil, nil, 16000, 5.0)

	if name1 == name2 {
		t.Errorf("different speakers should get different names: %q == %q", name1, name2)
	}
	if name1 != "speaker_1" {
		t.Errorf("expected speaker_1, got %q", name1)
	}
	if name2 != "speaker_2" {
		t.Errorf("expected speaker_2, got %q", name2)
	}
}

func TestDiscoveryNoDiscover(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, true) // noDiscover=true

	name, _ := d.IdentifySpeaker(
		makeEmbedding(512, 1.0), nil, nil, 16000, 0.0,
	)
	if name != "Unknown_1" {
		t.Errorf("expected Unknown_1, got %q", name)
	}

	// Should NOT create a folder
	speakerDir := filepath.Join(dir, "Unknown_1")
	if _, err := os.Stat(speakerDir); err == nil {
		t.Error("should not create folder in noDiscover mode")
	}
}

func TestDiscoveryPersistence(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false) // noDiscover=false

	// Provide segAudio so persistence can store it
	segAudio := make([]float32, 16000) // 1 second silence
	for i := range segAudio {
		segAudio[i] = 0.01 // non-zero to avoid empty file
	}

	name, _ := d.IdentifySpeaker(
		makeEmbedding(512, 1.0), nil, segAudio, 16000, 1.5,
	)

	// Verify folder was created
	speakerDir := filepath.Join(dir, name)
	if _, err := os.Stat(speakerDir); err != nil {
		t.Fatalf("speaker folder not created: %v", err)
	}

	// Verify .profile.json exists
	profilePath := filepath.Join(speakerDir, ".profile.json")
	if _, err := os.Stat(profilePath); err != nil {
		t.Fatalf(".profile.json not created: %v", err)
	}

	// Verify audio sample was saved
	entries, _ := os.ReadDir(speakerDir)
	hasWav := false
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".wav" {
			hasWav = true
			break
		}
	}
	if !hasWav {
		t.Error("no audio sample .wav found in speaker folder")
	}
}

func TestDiscoveryNextIDFromExisting(t *testing.T) {
	dir := t.TempDir()

	// Pre-create speaker_3 folder
	os.MkdirAll(filepath.Join(dir, "speaker_3"), 0755)

	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})

	d := NewDiscovery(store, nil, matcher, 0.6, false)

	name, _ := d.IdentifySpeaker(
		makeEmbedding(512, 1.0), nil, nil, 16000, 0.0,
	)
	// Should start at speaker_4, not speaker_1
	if name != "speaker_4" {
		t.Errorf("expected speaker_4, got %q", name)
	}
}

// makeEmbedding creates a 512-dim embedding filled with the given value.
func makeEmbedding(dim int, value float32) []float32 {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = value
	}
	return emb
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/speaker/ -v -run TestDiscovery`
Expected: FAIL — `NewDiscovery` not defined

- [ ] **Step 3: Implement discovery.go**

Create `internal/speaker/discovery.go`:

```go
package speaker

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// unknownSpeaker represents a speaker discovered during the current session.
type unknownSpeaker struct {
	name      string
	embedding []float32
}

// Discovery manages unknown speaker detection and clustering within a
// single transcription session.
type Discovery struct {
	store      *Store
	extractor  *Extractor
	matcher    *Matcher
	threshold  float32
	noDiscover bool
	unknowns   []unknownSpeaker
	nextID     int
}

// NewDiscovery creates a Discovery manager.
// It scans the store for existing speaker_N directories to avoid ID collisions.
func NewDiscovery(store *Store, extractor *Extractor, matcher *Matcher, threshold float32, noDiscover bool) *Discovery {
	nextID := scanMaxSpeakerID(store.Root()) + 1
	return &Discovery{
		store:      store,
		extractor:  extractor,
		matcher:    matcher,
		threshold:  threshold,
		noDiscover: noDiscover,
		nextID:     nextID,
	}
}

// scanMaxSpeakerID finds the highest N in existing speaker_N directories.
// Returns 0 if none found.
func scanMaxSpeakerID(dir string) int {
	re := regexp.MustCompile(`^speaker_(\d+)$`)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	maxID := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		matches := re.FindStringSubmatch(e.Name())
		if len(matches) == 2 {
			if id, err := strconv.Atoi(matches[1]); err == nil && id > maxID {
				maxID = id
			}
		}
	}
	return maxID
}

// IdentifySpeaker matches an embedding against known profiles and
// session-local unknowns. If no match is found, creates a new unknown speaker.
//
// segAudio and sampleRate are used for persisting audio samples when
// noDiscover is false. They can be nil/0 if persistence is not needed.
func (d *Discovery) IdentifySpeaker(
	embedding []float32,
	profiles []types.SpeakerProfile,
	segAudio []float32,
	sampleRate int,
	segStart float64,
) (string, float32) {
	// 1. Try matching against registered profiles
	if len(profiles) > 0 {
		result := d.matcher.Match(embedding, profiles, d.threshold)
		if result.Name != "" {
			return result.Name, result.Similarity
		}
	}

	// 2. Try matching against unknowns discovered in this session
	for _, u := range d.unknowns {
		sim := CosineSimilarity(embedding, u.embedding)
		if sim >= d.threshold {
			return u.name, sim
		}
	}

	// 3. Create new unknown
	name := d.createUnknown(embedding, segAudio, sampleRate, segStart)
	return name, 0
}

// createUnknown registers a new unknown speaker and optionally persists it.
func (d *Discovery) createUnknown(embedding []float32, segAudio []float32, sampleRate int, segStart float64) string {
	id := d.nextID
	d.nextID++

	var name string
	if d.noDiscover {
		name = fmt.Sprintf("Unknown_%d", id)
	} else {
		name = fmt.Sprintf("speaker_%d", id)
		d.persistUnknown(name, embedding, segAudio, sampleRate, segStart)
	}

	d.unknowns = append(d.unknowns, unknownSpeaker{name: name, embedding: embedding})
	return name
}

// persistUnknown creates the speaker folder, saves audio sample, and writes .profile.json.
func (d *Discovery) persistUnknown(name string, embedding []float32, segAudio []float32, sampleRate int, segStart float64) {
	speakerDir := filepath.Join(d.store.Root(), name)
	os.MkdirAll(speakerDir, 0755)

	// Save audio segment if available
	if len(segAudio) > 0 && sampleRate > 0 {
		timestamp := fmt.Sprintf("%04d", int(segStart*100))
		wavPath := filepath.Join(speakerDir, fmt.Sprintf("auto_segment_%s.wav", timestamp))
		audio.WriteWAV(wavPath, segAudio, sampleRate)
	}

	// Build and save profile
	now := time.Now().Format(time.RFC3339)
	profile := types.SpeakerProfile{
		Name: name,
		Embeddings: []types.SampleEmbedding{
			{
				File:      fmt.Sprintf("auto_segment_%04d.wav", int(segStart*100)),
				Hash:      "",
				Embedding: embedding,
			},
		},
		Dim:       len(embedding),
		Model:     "campplus_sv_zh-cn",
		CreatedAt: now,
		UpdatedAt: now,
	}
	d.store.SaveProfile(profile)
}

// Unknowns returns the names of all unknown speakers found in this session.
func (d *Discovery) Unknowns() []string {
	names := make([]string, len(d.unknowns))
	for i, u := range d.unknowns {
		names[i] = u.name
	}
	return names
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./internal/speaker/ -v -run TestDiscovery`
Expected: All 7 TestDiscovery* tests PASS

- [ ] **Step 5: Run all tests to check for regressions**

Run: `go test ./... -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add internal/speaker/discovery.go internal/speaker/discovery_test.go
git commit -m "feat: unknown speaker auto-discovery with cross-segment clustering"
```

---

## Task 3: Transcribe Command — Discovery Integration

**Files:**
- Modify: `cmd/commands/transcribe.go`

- [ ] **Step 1: Replace direct matcher call with Discovery**

In `cmd/commands/transcribe.go`, replace lines 113-183 (the store/matcher creation + segment loop + speaker counting).

After step 11 (read WAV), replace the segment processing section:

```go
			// 12. Initialize Discovery for unknown speaker tracking
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})
			discovery := speaker.NewDiscovery(store, extractor, matcher, float32(cfg.Threshold), noDiscover)

			// 13. Process each ASR segment
			segments := make([]types.TranscriptSegment, 0, len(results))
			for _, r := range results {
				segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

				// Speaker identification via Discovery
				speakerName := "Unknown"
				var speakerConf float32
				if len(segAudio) > 0 {
					emb, embErr := extractor.Extract(segAudio, wavSampleRate)
					if embErr == nil {
						speakerName, speakerConf = discovery.IdentifySpeaker(
							emb, profiles, segAudio, wavSampleRate, r.Start,
						)
					}
				}

				// Emotion classification (unchanged)
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

			// 14. Build metadata with speaker counts
			speakerSet := make(map[string]bool)
			identified := 0
			for _, seg := range segments {
				speakerSet[seg.Speaker] = true
				if seg.Speaker != "Unknown" &&
					!strings.HasPrefix(seg.Speaker, "speaker_") &&
					!strings.HasPrefix(seg.Speaker, "Unknown_") {
					identified++
				}
			}
```

The metadata construction, output formatting, and helpers remain the same.

- [ ] **Step 2: Verify compilation**

Run: `go build ./cmd/main.go`
Expected: BUILD SUCCESS

- [ ] **Step 3: Run all tests**

Run: `go test ./...`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add cmd/commands/transcribe.go
git commit -m "feat: transcribe with unknown speaker auto-discovery"
```

---

## Task 4: Speakers Verify Command

**Files:**
- Modify: `cmd/commands/speakers.go`

- [ ] **Step 1: Implement the verify command**

Replace the `newSpeakersVerifyCmd` function in `cmd/commands/speakers.go`:

```go
func newSpeakersVerifyCmd() *cobra.Command {
	var name, audioPath string
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify speaker recognition accuracy",
		RunE: func(cmd *cobra.Command, args []string) error {
			// 1. Load config
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}

			// 2. Load speaker profile
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profile, err := store.LoadProfile(name)
			if err != nil {
				return fmt.Errorf("load profile: %w", err)
			}
			if profile == nil {
				return fmt.Errorf("speaker %q not found in %s", name, speakersDir)
			}
			if len(profile.Embeddings) == 0 {
				return fmt.Errorf("speaker %q has no embeddings (run enroll first)", name)
			}

			// 3. Extract embedded binaries (for ffmpeg)
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// 4. Ensure speaker model
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}

			// 5. Initialize extractor
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("create extractor: %w", err)
			}
			defer extractor.Close()

			// 6. Convert and read test audio
			tmpDir, err := os.MkdirTemp("", "met-verify-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			tempWav := filepath.Join(tmpDir, "verify.wav")
			if err := audio.ConvertToWAV(bins.FFmpeg, audioPath, tempWav); err != nil {
				return fmt.Errorf("convert audio: %w", err)
			}

			samples, sampleRate, err := audio.ReadWAV(tempWav)
			if err != nil {
				return fmt.Errorf("read audio: %w", err)
			}

			// 7. Extract test embedding
			testEmb, err := extractor.Extract(samples, sampleRate)
			if err != nil {
				return fmt.Errorf("extract embedding: %w", err)
			}

			// 8. Match against profile
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})
			result := matcher.Match(testEmb, []types.SpeakerProfile{*profile}, float32(cfg.Threshold))

			// 9. Output result
			fmt.Printf("Verifying against %s...\n", name)
			fmt.Printf("  Similarity: %.2f (threshold: %.2f)\n", result.Similarity, cfg.Threshold)
			if result.Name != "" {
				fmt.Printf("  Result: ✓ MATCH\n")
			} else {
				fmt.Printf("  Result: ✗ NO MATCH\n")
			}

			return nil
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "speaker name (required)")
	cmd.Flags().StringVar(&audioPath, "audio", "", "test audio file path (required)")
	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("audio")
	return cmd
}
```

Update the imports at the top of speakers.go:
```go
import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)
```

- [ ] **Step 2: Verify compilation**

Run: `go build ./cmd/main.go`
Expected: BUILD SUCCESS

- [ ] **Step 3: Verify help output**

Run: `go run ./cmd/main.go speakers verify --help`
Expected: Shows --name and --audio flags

- [ ] **Step 4: Run all tests**

Run: `go test ./...`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add cmd/commands/speakers.go
git commit -m "feat: speakers verify command for recognition accuracy testing"
```

---

## Task 5: Build + E2E Verification

- [ ] **Step 1: Run all tests**

```bash
go test ./... -v -count=1
```

Expected: All pass.

- [ ] **Step 2: Build binary**

```bash
go build -o /tmp/met-phase4 ./cmd/main.go
```

Expected: BUILD SUCCESS.

- [ ] **Step 3: Verify CLI help**

```bash
/tmp/met-phase4 transcribe --help
/tmp/met-phase4 speakers verify --help
```

Expected: transcribe shows `--no-discover` flag. verify shows `--name` and `--audio`.

- [ ] **Step 4: Cleanup**

```bash
rm /tmp/met-phase4
```

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: Phase 4 E2E verification fixes"
```

---

## Self-Review Checklist

- **Spec §2 (Discovery Module):** ✓ Task 2 — NewDiscovery, IdentifySpeaker, clustering, persistence, nextID scan
- **Spec §3 (speakers verify):** ✓ Task 4 — full verify command
- **Spec §4 (Transcribe Integration):** ✓ Task 3 — Discovery replaces direct matcher
- **Spec §5 (Store.Root):** ✓ Task 1
- **noDiscover behavior:** ✓ Task 2 test `TestDiscoveryNoDiscover` — Unknown_N naming, no folder
- **Cross-segment clustering:** ✓ Task 2 test `TestDiscoveryCrossSegmentClustering`
- **nextID from existing folders:** ✓ Task 2 test `TestDiscoveryNextIDFromExisting`
- **Persistence (folder + wav + profile):** ✓ Task 2 test `TestDiscoveryPersistence`
- **Metadata speaker counting:** ✓ Task 3 — identified count excludes speaker_N and Unknown_N
- **No placeholders:** All code blocks complete
- **Type consistency:** `Discovery`, `IdentifySpeaker`, `Unknowns` consistent across tasks
