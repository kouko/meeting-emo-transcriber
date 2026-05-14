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

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 0, 0}
	sim := CosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("zero vector: got %f, want 0", sim)
	}
}

func TestMaxSimilarityStrategy_BestMatch(t *testing.T) {
	strategy := &MaxSimilarityStrategy{}
	profile := types.SpeakerProfile{
		Name: "TestUser",
		Voiceprints: []types.Voiceprint{
			{Vector: []float32{1, 0, 0}},
			{Vector: []float32{0, 1, 0}},
			{Vector: []float32{0.7, 0.7, 0}},
		},
	}
	query := []float32{0.71, 0.71, 0}
	score := strategy.Score(query, profile)
	if score < 0.99 {
		t.Errorf("expected high score for close match, got %f", score)
	}
}

func TestMatcherMatch_AboveThreshold(t *testing.T) {
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 1, 0}}}},
	}
	result := matcher.Match([]float32{0.99, 0.1, 0}, profiles, 0.6)
	if result.Name != "Alice" {
		t.Errorf("expected Alice, got %q", result.Name)
	}
}

func TestMatcherMatch_BelowThreshold(t *testing.T) {
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	profiles := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}},
	}
	result := matcher.Match([]float32{0, 1, 0}, profiles, 0.6)
	if result.Name != "" {
		t.Errorf("expected empty name for no match, got %q", result.Name)
	}
}

func TestMatcherMatch_EmptyProfiles(t *testing.T) {
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	result := matcher.Match([]float32{1, 0, 0}, nil, 0.6)
	if result.Name != "" {
		t.Errorf("expected empty name for no profiles, got %q", result.Name)
	}
}

func TestBestSimilarity_PicksHighest(t *testing.T) {
	profile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Vector: []float32{0, 1, 0}}, // orthogonal
			{Vector: []float32{1, 0, 0}}, // matches
			{Vector: []float32{0, 0, 1}}, // orthogonal
		},
	}
	got := BestSimilarity([]float32{0.99, 0.1, 0}, profile)
	if got < 0.98 {
		t.Errorf("got %f, want >= 0.98 (closest voiceprint match)", got)
	}
}

func TestBestSimilarity_DimMismatchSkipped(t *testing.T) {
	profile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Vector: []float32{1, 0}},       // 2-d, will be skipped
			{Vector: []float32{1, 0, 0, 0}}, // 4-d, will be skipped
		},
	}
	// All voiceprints mismatch the 3-d query — best stays at -1.
	got := BestSimilarity([]float32{1, 0, 0}, profile)
	if got != -1 {
		t.Errorf("got %f, want -1 (no compatible voiceprint)", got)
	}
}

func TestBestSimilarity_EmptyProfile(t *testing.T) {
	got := BestSimilarity([]float32{1, 0, 0}, types.SpeakerProfile{})
	if got != -1 {
		t.Errorf("got %f, want -1 (no voiceprints)", got)
	}
}
