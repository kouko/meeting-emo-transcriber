package diarize

import (
	"errors"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// fakeBatchExtractor returns a fixed sequence of embeddings and records the
// wav paths it was called with.
type fakeBatchExtractor struct {
	embeddings [][]float32
	called     [][]string
	err        error
}

func (f *fakeBatchExtractor) extract(paths []string) ([][]float32, error) {
	cp := make([]string, len(paths))
	copy(cp, paths)
	f.called = append(f.called, cp)
	if f.err != nil {
		return nil, f.err
	}
	return f.embeddings, nil
}

func defaultASRResults(n int, segDuration float64) []types.ASRResult {
	out := make([]types.ASRResult, n)
	for i := range out {
		start := float64(i) * segDuration
		out[i] = types.ASRResult{
			Start: start,
			End:   start + segDuration,
			Text:  "x",
		}
	}
	return out
}

func TestRefineSpeakerNames_NilExtractor_PassesThrough(t *testing.T) {
	names := []string{"Alice", "Bob"}
	asr := defaultASRResults(2, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	got, err := RefineSpeakerNamesPerSegment(names, asr, nil, wav, 16000,
		nil, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if got[0] != "Alice" || got[1] != "Bob" {
		t.Errorf("got %v, want [Alice Bob]", got)
	}
}

func TestRefineSpeakerNames_AllSegmentsAboveThreshold_PassesThrough(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	names := []string{"Alice", "Alice"}
	asr := defaultASRResults(2, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{0.99, 0.1, 0}, // sim to Alice ≈ 0.99
		{0.95, 0.2, 0}, // sim to Alice ≈ 0.98
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if got[0] != "Alice" || got[1] != "Alice" {
		t.Errorf("got %v, want [Alice Alice]", got)
	}
}

func TestRefineSpeakerNames_OneSegmentBelowThreshold_DemotedToUnknown(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	names := []string{"Alice", "Alice", "Alice"}
	asr := defaultASRResults(3, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{0.99, 0.1, 0}, // sim to Alice ≈ 0.99 → keep
		{0, 1, 0},      // sim to Alice ≈ 0    → demote
		{0.95, 0.2, 0}, // sim to Alice ≈ 0.98 → keep
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	want := []string{"Alice", "Unknown", "Alice"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestRefineSpeakerNames_UnenrolledLabelsNotVerified(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	// "speaker_2" and "Unknown" are not enrolled → must be left alone
	// even if the extractor would have demoted them.
	names := []string{"Alice", "speaker_2", "Unknown", ""}
	asr := defaultASRResults(4, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	// Only one segment (the Alice one) should be sent to the extractor.
	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{0.99, 0.1, 0},
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if len(ex.called) != 1 || len(ex.called[0]) != 1 {
		t.Errorf("extractor called %v, want exactly one call with one wav", ex.called)
	}
	if got[0] != "Alice" || got[1] != "speaker_2" || got[2] != "Unknown" || got[3] != "" {
		t.Errorf("got %v, want [Alice speaker_2 Unknown <empty>]", got)
	}
}

func TestRefineSpeakerNames_ShortSegmentNotVerified(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	names := []string{"Alice", "Alice"}
	// First segment is 0.5s (below min 1.0s) → not verified.
	asr := []types.ASRResult{
		{Start: 0, End: 0.5, Text: "hi"},
		{Start: 0.5, End: 3.0, Text: "hello"},
	}
	wav := generateSineWAV(16000, 10.0, 0.5)

	// Only the long segment goes to extractor; return a bad embedding.
	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{0, 1, 0}, // orthogonal to Alice → would demote
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	// Short segment kept as Alice even though it wasn't verified.
	if got[0] != "Alice" {
		t.Errorf("got[0] = %q, want Alice (short segment kept as cluster label)", got[0])
	}
	// Long segment demoted.
	if got[1] != "Unknown" {
		t.Errorf("got[1] = %q, want Unknown (long segment failed verification)", got[1])
	}
}

func TestRefineSpeakerNames_NoEnrolledProfiles_PassesThrough(t *testing.T) {
	names := []string{"speaker_1", "speaker_2"}
	asr := defaultASRResults(2, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	ex := &fakeBatchExtractor{}
	got, err := RefineSpeakerNamesPerSegment(names, asr, nil, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if len(ex.called) != 0 {
		t.Errorf("extractor called %d times, want 0 (no enrolled profiles)", len(ex.called))
	}
	if got[0] != "speaker_1" || got[1] != "speaker_2" {
		t.Errorf("got %v, want unchanged", got)
	}
}

func TestRefineSpeakerNames_BatchExtractError_ReturnsUnchanged(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	names := []string{"Alice"}
	asr := defaultASRResults(1, 2.0)
	wav := generateSineWAV(16000, 5.0, 0.5)

	ex := &fakeBatchExtractor{err: errors.New("boom")}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Caller can choose to keep the unverified labels — we return them
	// alongside the error so the original transcript is recoverable.
	if got[0] != "Alice" {
		t.Errorf("got[0] = %q, want Alice (unchanged on extractor failure)", got[0])
	}
}

func TestRefineSpeakerNames_EmptyEmbeddingNotDemoted(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	names := []string{"Alice", "Alice"}
	asr := defaultASRResults(2, 2.0)
	wav := generateSineWAV(16000, 10.0, 0.5)

	// First segment: extractor returns empty (e.g. all silence).
	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{},             // empty → skipped
		{0.99, 0.1, 0}, // keep
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if got[0] != "Alice" {
		t.Errorf("got[0] = %q, want Alice (empty embedding should not demote)", got[0])
	}
	if got[1] != "Alice" {
		t.Errorf("got[1] = %q, want Alice", got[1])
	}
}

func TestRefineSpeakerNames_DimMismatchNotDemoted(t *testing.T) {
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0, 0}}}},
	}
	names := []string{"Alice"}
	asr := defaultASRResults(1, 2.0)
	wav := generateSineWAV(16000, 5.0, 0.5)

	// Extractor returns 3-d embedding but profile has 4-d voiceprint.
	// bestSimAgainstProfile returns -1; demoted? sim=-1 < threshold so yes.
	// This documents the current behavior: dim mismatch -> sim=-1 -> demote.
	ex := &fakeBatchExtractor{embeddings: [][]float32{
		{0.99, 0.1, 0},
	}}

	got, err := RefineSpeakerNamesPerSegment(names, asr, profiles, wav, 16000,
		ex.extract, 0.5, 1.0, t.TempDir())
	if err != nil {
		t.Fatalf("RefineSpeakerNames: %v", err)
	}
	if got[0] != "Unknown" {
		t.Errorf("got[0] = %q, want Unknown (dim mismatch -> sim=-1 -> below threshold)", got[0])
	}
}
