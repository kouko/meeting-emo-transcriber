# Speaker Diarization Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-segment embedding matching (296 speakers for 4 people) with sherpa-onnx OfflineSpeakerDiarization (Pyannote segmentation + FastClustering), producing accurate speaker counts and enrolled name matching.

**Architecture:** Two parallel pipelines — Whisper ASR produces text+timestamps, sherpa-onnx diarization produces speaker_ID+time_ranges — merged by time overlap. Enrolled profiles matched via cluster-level embedding extraction. New `internal/diarize/` package wraps sherpa-onnx API.

**Tech Stack:** Go 1.25, sherpa-onnx-go-macos (OfflineSpeakerDiarization), Pyannote segmentation-3.0 ONNX, ERes2Net embedding ONNX

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `internal/diarize/diarize.go` | Diarizer wrapper around sherpa-onnx OfflineSpeakerDiarization |
| `internal/diarize/merge.go` | AssignSpeakers (time overlap) + ResolveSpeakerNames (enrolled profile matching) |
| `internal/diarize/diarize_test.go` | Unit tests for merge logic |

### Modified files

| File | Changes |
|------|---------|
| `internal/models/registry.go` | Add pyannote-segmentation-3-0 + eres2net embedding models |
| `cmd/commands/transcribe.go` | Replace discovery pipeline with diarization pipeline |

### Deleted files

| File | Reason |
|------|--------|
| `internal/speaker/discovery.go` | Replaced by diarize package |
| `internal/speaker/discovery_test.go` | Corresponding tests |

---

## Task 1: Add Diarization Models to Registry

**Files:**
- Modify: `internal/models/registry.go`
- Modify: `internal/models/models_test.go`

- [ ] **Step 1: Add model entries to registry**

Add to `internal/models/registry.go` Registry map:

```go
"pyannote-segmentation-3-0": {
    Name:      "pyannote-segmentation-3-0",
    URL:       "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    SHA256:    "",
    Size:      5000000,
    Category:  "diarize",
    IsArchive: true,
},
"eres2net-embedding": {
    Name:     "eres2net-embedding",
    URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    SHA256:   "",
    Size:     50000000,
    Category: "diarize",
},
```

- [ ] **Step 2: Add tests**

Add to `internal/models/models_test.go`:

```go
func TestRegistryContainsDiarizeModels(t *testing.T) {
    for _, name := range []string{"pyannote-segmentation-3-0", "eres2net-embedding"} {
        t.Run(name, func(t *testing.T) {
            info, ok := Registry[name]
            if !ok {
                t.Fatalf("Registry missing %q", name)
            }
            if info.Category != "diarize" {
                t.Errorf("Category = %q, want \"diarize\"", info.Category)
            }
            if info.URL == "" {
                t.Error("URL is empty")
            }
        })
    }
}
```

- [ ] **Step 3: Run tests**

Run: `go test ./internal/models/ -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add internal/models/
git commit -m "feat: add pyannote segmentation and eres2net models to registry"
```

---

## Task 2: Diarizer Wrapper

**Files:**
- Create: `internal/diarize/diarize.go`

- [ ] **Step 1: Create diarize.go**

```go
package diarize

import (
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
)

// Segment represents a speaker diarization result.
type Segment struct {
	Start   float64
	End     float64
	Speaker int // 0-indexed cluster ID
}

// Diarizer wraps sherpa-onnx OfflineSpeakerDiarization.
type Diarizer struct {
	inner *sherpa.OfflineSpeakerDiarization
}

// NewDiarizer creates a diarizer.
// segModelDir: path to directory containing pyannote model.onnx
// embModelPath: path to speaker embedding .onnx file
// numClusters: if > 0, use fixed cluster count; otherwise use threshold
// threshold: clustering threshold (higher = fewer clusters); ignored if numClusters > 0
func NewDiarizer(segModelDir, embModelPath string, threads int, numClusters int, threshold float32) (*Diarizer, error) {
	config := &sherpa.OfflineSpeakerDiarizationConfig{}

	// Segmentation model (Pyannote)
	config.Segmentation.Pyannote.Model = filepath.Join(segModelDir, "model.onnx")
	config.Segmentation.NumThreads = threads
	config.Segmentation.Provider = "cpu"

	// Embedding model (ERes2Net)
	config.Embedding.Model = embModelPath
	config.Embedding.NumThreads = threads
	config.Embedding.Provider = "cpu"

	// Clustering
	if numClusters > 0 {
		config.Clustering.NumClusters = numClusters
	} else {
		config.Clustering.Threshold = threshold
	}

	inner := sherpa.NewOfflineSpeakerDiarization(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create diarizer (check model paths: seg=%s, emb=%s)", segModelDir, embModelPath)
	}

	return &Diarizer{inner: inner}, nil
}

// Process runs diarization on audio samples (must be at SampleRate() Hz).
// Returns segments sorted by start time.
func (d *Diarizer) Process(samples []float32) []Segment {
	raw := d.inner.Process(samples)
	segments := make([]Segment, len(raw))
	for i, r := range raw {
		segments[i] = Segment{
			Start:   float64(r.Start),
			End:     float64(r.End),
			Speaker: r.Speaker,
		}
	}
	return segments
}

// SampleRate returns the expected audio sample rate (typically 16000).
func (d *Diarizer) SampleRate() int {
	return d.inner.SampleRate()
}

// Close releases resources.
func (d *Diarizer) Close() {
	if d.inner != nil {
		sherpa.DeleteOfflineSpeakerDiarization(d.inner)
		d.inner = nil
	}
}
```

Add `"path/filepath"` to imports.

- [ ] **Step 2: Verify compilation**

Run: `go build ./internal/diarize/`
Expected: BUILD SUCCESS

- [ ] **Step 3: Commit**

```bash
git add internal/diarize/diarize.go
git commit -m "feat: diarizer wrapper for sherpa-onnx OfflineSpeakerDiarization"
```

---

## Task 3: Merge Logic (AssignSpeakers + ResolveSpeakerNames)

**Files:**
- Create: `internal/diarize/merge.go`
- Create: `internal/diarize/diarize_test.go`

- [ ] **Step 1: Write tests for AssignSpeakers**

Create `internal/diarize/diarize_test.go`:

```go
package diarize

import (
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestAssignSpeakersExactOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 0.0, End: 5.0, Text: "Hello"},
		{Start: 5.0, End: 10.0, Text: "World"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 5.0, Speaker: 0},
		{Start: 5.0, End: 10.0, Speaker: 1},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != 0 {
		t.Errorf("segment 0: got speaker %d, want 0", ids[0])
	}
	if ids[1] != 1 {
		t.Errorf("segment 1: got speaker %d, want 1", ids[1])
	}
}

func TestAssignSpeakersPartialOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 2.0, End: 8.0, Text: "Hello"},
	}
	// Speaker 0: 0-5, Speaker 1: 5-10
	// ASR 2-8 overlaps with speaker_0 for 3s (2-5) and speaker_1 for 3s (5-8)
	// Tie → first match wins (speaker 0)
	diarSegments := []Segment{
		{Start: 0.0, End: 5.0, Speaker: 0},
		{Start: 5.0, End: 10.0, Speaker: 1},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != 0 {
		t.Errorf("got speaker %d, want 0 (first match on tie)", ids[0])
	}
}

func TestAssignSpeakersNoOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 20.0, End: 25.0, Text: "Late"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 10.0, Speaker: 0},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != -1 {
		t.Errorf("got speaker %d, want -1 (no overlap)", ids[0])
	}
}

func TestAssignSpeakersEmpty(t *testing.T) {
	ids := AssignSpeakers(nil, nil)
	if len(ids) != 0 {
		t.Errorf("expected empty, got %v", ids)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/diarize/ -v`
Expected: FAIL — AssignSpeakers not defined

- [ ] **Step 3: Implement merge.go**

Create `internal/diarize/merge.go`:

```go
package diarize

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// AssignSpeakers maps each ASR result to a diarization speaker by maximum time overlap.
// Returns a slice of speaker IDs (0-indexed) parallel to asrResults.
// Returns -1 for segments with no diarization overlap.
func AssignSpeakers(asrResults []types.ASRResult, diarSegments []Segment) []int {
	ids := make([]int, len(asrResults))
	for i, asr := range asrResults {
		bestSpeaker := -1
		var bestOverlap float64
		for _, seg := range diarSegments {
			overlap := overlapDuration(asr.Start, asr.End, seg.Start, seg.End)
			if overlap > bestOverlap {
				bestOverlap = overlap
				bestSpeaker = seg.Speaker
			}
		}
		ids[i] = bestSpeaker
	}
	return ids
}

// overlapDuration computes the overlap in seconds between two time intervals.
func overlapDuration(s1, e1, s2, e2 float64) float64 {
	start := math.Max(s1, s2)
	end := math.Min(e1, e2)
	if end > start {
		return end - start
	}
	return 0
}

// ResolveSpeakerNames maps cluster IDs to human-readable names.
// For each unique cluster, extracts representative audio, computes embedding,
// and matches against enrolled profiles.
// Unmatched clusters → "speaker_N" with auto-created directories.
func ResolveSpeakerNames(
	speakerIDs []int,
	diarSegments []Segment,
	wavSamples []float32,
	sampleRate int,
	extractor *speaker.Extractor,
	profiles []types.SpeakerProfile,
	matcher *speaker.Matcher,
	threshold float32,
	store *speaker.Store,
) ([]string, error) {
	// Find unique cluster IDs
	clusterSet := make(map[int]bool)
	for _, id := range speakerIDs {
		if id >= 0 {
			clusterSet[id] = true
		}
	}

	// For each cluster, find the longest segment as representative
	clusterNames := make(map[int]string)
	nextUnknownID := scanMaxSpeakerID(store.Root()) + 1

	for clusterID := range clusterSet {
		// Find longest diarization segment for this cluster
		var bestSeg Segment
		var bestLen float64
		for _, seg := range diarSegments {
			if seg.Speaker == clusterID {
				segLen := seg.End - seg.Start
				if segLen > bestLen {
					bestLen = segLen
					bestSeg = seg
				}
			}
		}

		// Extract representative audio
		segAudio := audio.ExtractSegment(wavSamples, sampleRate, bestSeg.Start, bestSeg.End)

		// Try to match against enrolled profiles
		name := ""
		if len(segAudio) > 0 && extractor != nil && len(profiles) > 0 {
			emb, err := extractor.Extract(segAudio, sampleRate)
			if err == nil {
				result := matcher.Match(emb, profiles, threshold)
				if result.Name != "" {
					name = result.Name
				}
			}
		}

		if name == "" {
			// Unmatched → create speaker_N
			name = fmt.Sprintf("speaker_%d", nextUnknownID)
			nextUnknownID++

			// Auto-create directory and save profile
			persistUnknownSpeaker(store, name, segAudio, sampleRate, extractor)
		}

		clusterNames[clusterID] = name
	}

	// Map IDs to names
	result := make([]string, len(speakerIDs))
	for i, id := range speakerIDs {
		if id >= 0 {
			result[i] = clusterNames[id]
		} else {
			result[i] = "Unknown"
		}
	}
	return result, nil
}

// persistUnknownSpeaker saves an auto-discovered speaker to the speakers directory.
func persistUnknownSpeaker(store *speaker.Store, name string, segAudio []float32, sampleRate int, extractor *speaker.Extractor) {
	speakerDir := filepath.Join(store.Root(), name)
	os.MkdirAll(speakerDir, 0755)

	// Save audio sample
	wavPath := filepath.Join(speakerDir, "auto_sample.wav")
	audio.WriteWAV(wavPath, segAudio, sampleRate)

	// Compute and save embedding
	var embeddings []types.SampleEmbedding
	if extractor != nil && len(segAudio) > 0 {
		emb, err := extractor.Extract(segAudio, sampleRate)
		if err == nil {
			embeddings = append(embeddings, types.SampleEmbedding{
				File:      "auto_sample.wav",
				Embedding: emb,
			})
		}
	}

	now := time.Now().Format(time.RFC3339)
	profile := types.SpeakerProfile{
		Name:       name,
		Embeddings: embeddings,
		Dim:        len(embeddings[0].Embedding),
		Model:      "eres2net_base",
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	store.SaveProfile(profile)
}

// scanMaxSpeakerID finds the highest N in existing speaker_N directories.
func scanMaxSpeakerID(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	maxID := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		var id int
		if _, err := fmt.Sscanf(e.Name(), "speaker_%d", &id); err == nil && id > maxID {
			maxID = id
		}
	}
	return maxID
}
```

- [ ] **Step 4: Run tests**

Run: `go test ./internal/diarize/ -v`
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add internal/diarize/
git commit -m "feat: ASR-diarization merge with enrolled profile matching"
```

---

## Task 4: Refactor Transcribe Command

**Files:**
- Modify: `cmd/commands/transcribe.go`

- [ ] **Step 1: Replace the speaker identification pipeline**

Rewrite `cmd/commands/transcribe.go`. Key changes:
- Remove `--no-discover` flag, add `--num-speakers` flag
- Replace discovery pipeline (lines 95-178) with diarization pipeline
- Add new imports: `"github.com/kouko/meeting-emo-transcriber/internal/diarize"`
- Remove import: discovery is no longer used

The new transcribe command RunE body (replace everything from step 8 onwards):

```go
			// 8. Ensure diarization models
			fmt.Fprintf(os.Stderr, "[6/8] Running speaker diarization...\n")
			segModelDir, err := models.EnsureModel("pyannote-segmentation-3-0")
			if err != nil {
				return fmt.Errorf("ensure segmentation model: %w", err)
			}
			embModelPath, err := models.EnsureModel("eres2net-embedding")
			if err != nil {
				return fmt.Errorf("ensure embedding model: %w", err)
			}

			// 9. Read full WAV for diarization + segment extraction
			wavSamples, wavSampleRate, err := audio.ReadWAV(tempWavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}

			// 10. Run diarization
			diarizer, err := diarize.NewDiarizer(segModelDir, embModelPath, cfg.Threads, numSpeakers, threshold)
			if err != nil {
				return fmt.Errorf("init diarizer: %w", err)
			}
			defer diarizer.Close()

			diarSegments := diarizer.Process(wavSamples)

			// 11. Assign speakers to ASR segments
			speakerIDs := diarize.AssignSpeakers(results, diarSegments)

			// 12. Resolve speaker names (match enrolled profiles)
			fmt.Fprintf(os.Stderr, "[7/8] Resolving speaker identities...\n")
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init speaker extractor: %w", err)
			}
			defer extractor.Close()

			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})

			speakerNames, err := diarize.ResolveSpeakerNames(
				speakerIDs, diarSegments, wavSamples, wavSampleRate,
				extractor, profiles, matcher, float32(cfg.Threshold), store,
			)
			if err != nil {
				return fmt.Errorf("resolve speaker names: %w", err)
			}

			// 13. Emotion classification + build segments
			emotionModelDir, err := models.EnsureModel("sensevoice-small-int8")
			if err != nil {
				return fmt.Errorf("ensure emotion model: %w", err)
			}
			classifier, err := emotion.NewClassifier(emotionModelDir, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init emotion classifier: %w", err)
			}
			defer classifier.Close()

			segments := make([]types.TranscriptSegment, 0, len(results))
			for i, r := range results {
				segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

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
					Speaker:    speakerNames[i],
					Emotion:    emotionInfo,
					AudioEvent: audioEvent,
					Language:   r.Language,
					Text:       r.Text,
					Confidence: types.Confidence{Speaker: 0, Emotion: emotionConf},
				})
			}
```

Update flags section — remove `--no-discover`, add `--num-speakers`:

```go
	var (
		inputPath   string
		outputPath  string
		format      string
		language    string
		threshold   float32
		numSpeakers int
	)
	// ...
	cmd.Flags().Float32Var(&threshold, "threshold", 0.5, "diarization clustering threshold (higher = fewer speakers)")
	cmd.Flags().IntVar(&numSpeakers, "num-speakers", 0, "expected number of speakers (0 = auto-detect)")
	// Remove: cmd.Flags().BoolVar(&noDiscover, ...)
```

Update progress steps from [1/7]...[7/7] to [1/8]...[8/8].

Update metadata section to count speakers properly:

```go
			speakerSet := make(map[string]bool)
			identified := 0
			for _, seg := range segments {
				speakerSet[seg.Speaker] = true
				if !strings.HasPrefix(seg.Speaker, "speaker_") && seg.Speaker != "Unknown" {
					identified++
				}
			}
```

- [ ] **Step 2: Update imports**

Add: `"github.com/kouko/meeting-emo-transcriber/internal/diarize"`
Remove the `speaker.NewDiscovery` reference (no longer exists after deletion).

- [ ] **Step 3: Verify compilation**

Run: `go build ./cmd/main.go`
Expected: BUILD SUCCESS

- [ ] **Step 4: Run all tests**

Run: `go test ./...`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add cmd/commands/transcribe.go
git commit -m "refactor: replace per-segment discovery with diarization pipeline"
```

---

## Task 5: Delete Old Discovery Module

**Files:**
- Delete: `internal/speaker/discovery.go`
- Delete: `internal/speaker/discovery_test.go`

- [ ] **Step 1: Delete files**

```bash
rm internal/speaker/discovery.go internal/speaker/discovery_test.go
```

- [ ] **Step 2: Verify no references remain**

Run: `grep -r "discovery\|Discovery\|NewDiscovery" --include="*.go" . | grep -v _test.go | grep -v docs/`
Expected: No Go source files reference Discovery.

- [ ] **Step 3: Run all tests**

Run: `go test ./...`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add internal/speaker/
git commit -m "refactor: remove old per-segment discovery module"
```

---

## Task 6: Build + E2E Verification

- [ ] **Step 1: Run all tests**

```bash
go test ./... -v -count=1
```

Expected: All pass.

- [ ] **Step 2: Build binary**

```bash
go build -o metr ./cmd/main.go
```

Expected: BUILD SUCCESS.

- [ ] **Step 3: Verify CLI help**

```bash
./metr transcribe --help
```

Expected: Shows `--num-speakers` flag, no `--no-discover`.

- [ ] **Step 4: Cleanup test binary**

```bash
rm metr
```

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git diff --cached --stat
git commit -m "chore: diarization refactor E2E verification"
```

---

## Self-Review Checklist

- **Spec §2 (New Architecture):** ✓ Task 4 — parallel ASR + diarization pipelines
- **Spec §3 (diarize package):** ✓ Task 2 (Diarizer) + Task 3 (merge)
- **Spec §4 (Models):** ✓ Task 1 — pyannote + eres2net in registry
- **Spec §5 (Delete discovery):** ✓ Task 5
- **Spec §5 (CLI flags):** ✓ Task 4 — remove --no-discover, add --num-speakers
- **Spec §6 (Enrolled matching):** ✓ Task 3 — ResolveSpeakerNames
- **Spec §6 (Auto-create dirs):** ✓ Task 3 — persistUnknownSpeaker
- **No placeholders:** All code blocks complete
- **Type consistency:** `Segment`, `AssignSpeakers`, `ResolveSpeakerNames`, `Diarizer` consistent
