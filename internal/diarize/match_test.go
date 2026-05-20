package diarize

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// --- matchAgainstProfilesDetailed -----------------------------------------

func TestMatchAgainstProfilesDetailed_EmptyProfiles(t *testing.T) {
	res := matchAgainstProfilesDetailed([]float32{1, 0, 0}, nil, 0.5)
	if res.Matched {
		t.Errorf("Matched = true, want false (no profiles)")
	}
	if res.Name != "" {
		t.Errorf("Name = %q, want empty", res.Name)
	}
	if len(res.Details) != 0 {
		t.Errorf("Details len = %d, want 0", len(res.Details))
	}
}

func TestMatchAgainstProfilesDetailed_SingleProfileAboveThreshold(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	res := matchAgainstProfilesDetailed([]float32{0.99, 0.1, 0}, profiles, 0.6)
	if !res.Matched {
		t.Errorf("Matched = false, want true (sim >> 0.6)")
	}
	if res.Name != "Alice" {
		t.Errorf("Name = %q, want Alice", res.Name)
	}
	if res.BestSim < 0.99 {
		t.Errorf("BestSim = %f, want > 0.99", res.BestSim)
	}
}

func TestMatchAgainstProfilesDetailed_SingleProfileBelowThreshold(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	res := matchAgainstProfilesDetailed([]float32{0, 1, 0}, profiles, 0.5)
	if res.Matched {
		t.Errorf("Matched = true, want false (orthogonal)")
	}
	// Name is set to argmax even on miss so callers can diagnose;
	// the Matched flag is what callers must gate on.
	if res.Name != "Alice" {
		t.Errorf("Name = %q, want Alice (argmax even when below threshold)", res.Name)
	}
	if len(res.Details) != 1 || res.Details[0].ProfileName != "Alice" {
		t.Errorf("Details = %v, want one entry for Alice", res.Details)
	}
}

func TestMatchAgainstProfilesDetailed_MultipleProfiles_BestWins(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 1, 0}}}},
		{Name: "Carol", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 0, 1}}}},
	}
	// Vector closer to Bob.
	res := matchAgainstProfilesDetailed([]float32{0.1, 0.99, 0.1}, profiles, 0.5)
	if res.Name != "Bob" {
		t.Errorf("Name = %q, want Bob", res.Name)
	}
	if len(res.Details) != 3 {
		t.Errorf("Details len = %d, want 3", len(res.Details))
	}
	// Runner-up should be set to the second-highest profile score (Alice or
	// Carol, both ≈0.0995 — orthogonal except for the 0.1 axis components).
	if res.RunnerUpSim <= 0 || res.RunnerUpSim >= res.BestSim {
		t.Errorf("RunnerUpSim = %f, want 0 < runner-up < best (%f)", res.RunnerUpSim, res.BestSim)
	}
}

func TestMatchAgainstProfilesDetailed_SingleProfile_RunnerUpIsNegOne(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	res := matchAgainstProfilesDetailed([]float32{0.99, 0.1, 0}, profiles, 0.5)
	if res.RunnerUpSim != -1 {
		t.Errorf("RunnerUpSim = %f, want -1 (no second profile)", res.RunnerUpSim)
	}
}

// LOCK-IN: With max-similarity strategy, the highest-sim voiceprint inside any
// profile wins for that profile. If a profile contains multiple voiceprints
// (centroid + merged), the most-similar one is what counts.
func TestMatchAgainstProfilesDetailed_MaxAcrossVoiceprintsInProfile(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{
			Name: "Alice",
			Voiceprints: []types.Voiceprint{
				{Type: "centroid", Vector: []float32{0, 1, 0}}, // far
				{Type: "merged", Vector: []float32{1, 0, 0}},   // close
			},
		},
	}
	res := matchAgainstProfilesDetailed([]float32{0.99, 0.1, 0}, profiles, 0.6)
	if !res.Matched {
		t.Error("expected match (one voiceprint is very close)")
	}
	if res.Details[0].NumVoiceprints != 2 {
		t.Errorf("NumVoiceprints = %d, want 2", res.Details[0].NumVoiceprints)
	}
}

// LOCK-IN: Voiceprints whose dimension differs from the query are silently
// skipped (no warning, no error). This will become important when models change.
func TestMatchAgainstProfilesDetailed_DimMismatchSkipped(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{
			Name: "Alice",
			Voiceprints: []types.Voiceprint{
				{Vector: []float32{1, 0}},       // wrong dim
				{Vector: []float32{1, 0, 0}},    // right dim
			},
		},
	}
	res := matchAgainstProfilesDetailed([]float32{0.99, 0.1, 0}, profiles, 0.6)
	if !res.Matched {
		t.Error("expected match against the matching-dim voiceprint")
	}
	// The 2-d voiceprint must NOT contribute a score.
	if res.BestSim < 0.9 {
		t.Errorf("BestSim = %f, want >0.9 from the 3-d voiceprint", res.BestSim)
	}
}

// --- ResolveSpeakerNames with enrolled profiles ---------------------------

func newTestStore(t *testing.T) *speaker.Store {
	t.Helper()
	dir := t.TempDir()
	return speaker.NewStore(dir, config.SupportedAudioExtensions())
}

func TestResolveSpeakerNames_SingleClusterMatchesProfile(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	// Enrolled Alice: voiceprint = unit vector along x.
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0.99, 0.1, 0}, // very close to Alice
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "Alice" {
		t.Errorf("names[0] = %q, want Alice", names[0])
	}
}

func TestResolveSpeakerNames_BelowThresholdCreatesNewSpeaker(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0, 1, 0}, // orthogonal to Alice → sim = 0
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "speaker_1" {
		t.Errorf("names[0] = %q, want speaker_1 (no match → new)", names[0])
	}
	// speaker_1 dir must be created on disk.
	if _, err := os.Stat(filepath.Join(store.Root(), "speaker_1")); os.IsNotExist(err) {
		t.Error("speaker_1 directory should be created")
	}
}

// One-to-one assignment: when two clusters both match an enrolled profile,
// the higher-similarity cluster wins the name and the other falls through
// to a new speaker_N. This prevents Alice from appearing twice in the
// transcript when diarization split her into two clusters.
func TestResolveSpeakerNames_TwoClustersBothMatchAlice_OnlyHighestWinsName(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 60.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	// C1 has slightly higher cosine to Alice than C2, so C1 should win.
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 25, Speaker: "C1"},
			{Start: 25, End: 50, Speaker: "C2"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0.99, 0.1, 0}, // sim to Alice ≈ 0.985
			"C2": {0.95, 0.2, 0}, // sim to Alice ≈ 0.978
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1", "C2"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "Alice" {
		t.Errorf("names[0] = %q, want Alice (highest cosine)", names[0])
	}
	if names[1] != "speaker_1" {
		t.Errorf("names[1] = %q, want speaker_1 (Alice taken by C1)", names[1])
	}
	// speaker_1 dir must exist on disk for the displaced cluster.
	if _, err := os.Stat(filepath.Join(store.Root(), "speaker_1")); os.IsNotExist(err) {
		t.Error("speaker_1 directory should be created for displaced cluster")
	}
}

// Margin guard: when the best and runner-up profiles are too similar, the
// match is treated as ambiguous and the cluster is reassigned to a fresh
// speaker_N rather than gambling on the argmax.
func TestResolveSpeakerNames_AmbiguousMatch_RejectedByMargin(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0.95, 0.31, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			// Centroid sits between Alice and Bob: both score ≈0.987-0.989.
			// Margin (best - runner-up) ≈ 0.002, well below 0.07.
			"C1": {0.98, 0.16, 0},
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "speaker_1" {
		t.Errorf("names[0] = %q, want speaker_1 (margin guard rejects ambiguous match)", names[0])
	}
}

// Learn mode writes review samples under <root>/_metr/review/ rather than at
// the root, so subsequent runs do not pick them up as new enrolled speakers.
func TestResolveSpeakerNames_LearnMode_WritesToReviewSubdir(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0.99, 0.1, 0},
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", true, // learn=true
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "Alice" {
		t.Fatalf("names[0] = %q, want Alice", names[0])
	}

	// The review directory must exist under _metr/review/, not at root.
	reviewParent := filepath.Join(store.Root(), "_metr", "review")
	entries, err := os.ReadDir(reviewParent)
	if err != nil {
		t.Fatalf("expected _metr/review/ to be created: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("_metr/review/ is empty; learn-mode output not written")
	}
	foundReview := false
	for _, e := range entries {
		if e.IsDir() && strings.HasPrefix(e.Name(), "speaker_") && strings.Contains(e.Name(), "_match_Alice") {
			foundReview = true
			break
		}
	}
	if !foundReview {
		t.Errorf("no speaker_*_match_Alice dir under _metr/review/; got entries: %v", entries)
	}

	// The review dir must NOT appear at root (would pollute Store.List).
	rootEntries, _ := os.ReadDir(store.Root())
	for _, e := range rootEntries {
		if strings.Contains(e.Name(), "_match_") {
			t.Errorf("learn-mode review leaked to root: %q", e.Name())
		}
	}
	// And Store.List must NOT include _metr.
	listed, _ := store.List()
	for _, name := range listed {
		if name == "_metr" || strings.HasPrefix(name, "_") {
			t.Errorf("Store.List returned reserved entry %q", name)
		}
	}
}

// Margin disabled: with margin=0, the old greedy argmax behavior is recovered.
// Useful as an escape hatch for users who prefer the previous behavior.
func TestResolveSpeakerNames_MarginZero_AcceptsAmbiguous(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0.95, 0.31, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0.98, 0.16, 0},
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.0, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "Alice" && names[0] != "Bob" {
		t.Errorf("names[0] = %q, want Alice or Bob (margin disabled)", names[0])
	}
}

// Threshold semantics: at an explicit threshold of 0.55, a cosine of ≈0.6
// still matches. (The default threshold was raised from 0.55 to 0.65 in
// Step 1, but callers can still pass any value.)
func TestResolveSpeakerNames_ExplicitThreshold055_AcceptsBorderlineMatch(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			// Cosine to (1,0,0) ≈ 0.6
			"C1": {0.6, 0.8, 0},
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.55, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "Alice" {
		t.Errorf("names[0] = %q, want Alice (sim ≈0.6 > threshold 0.55)", names[0])
	}
}

// New default 0.65 rejects borderline cosine ≈0.60 matches that would have
// been accepted under the old 0.55 default.
func TestResolveSpeakerNames_DefaultThreshold065_RejectsBorderline(t *testing.T) {
	store := newTestStore(t)
	sampleRate := 16000
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{
			"C1": {0.6, 0.8, 0}, // cosine to Alice ≈ 0.6
		},
	}

	names, err := ResolveSpeakerNames(
		[]string{"C1"}, diarResult, wavSamples, sampleRate,
		profiles, 0.65, 0.07, store, "", false,
		15.0, 0.01,
		true, // discover
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}
	if names[0] != "speaker_1" {
		t.Errorf("names[0] = %q, want speaker_1 (sim ≈0.6 < new default threshold 0.65)", names[0])
	}
}

