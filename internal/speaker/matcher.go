package speaker

import (
	"math"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
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

// BestSimilarity returns the highest cosine similarity between emb and any
// voiceprint vector in profile. Voiceprints with mismatched dimensions are
// silently skipped (a different model may have produced the stored vector).
// Returns -1 when no compatible voiceprint exists.
//
// This is the canonical helper used by both cluster-level matching and
// per-segment re-verification — keeping a single implementation here
// prevents subtle drift between the two scoring paths.
func BestSimilarity(emb []float32, profile types.SpeakerProfile) float32 {
	var best float32 = -1
	for _, vp := range profile.Voiceprints {
		if len(vp.Vector) != len(emb) {
			continue
		}
		sim := CosineSimilarity(emb, vp.Vector)
		if sim > best {
			best = sim
		}
	}
	return best
}

type MatchStrategy interface {
	Score(segmentEmb []float32, profile types.SpeakerProfile) float32
}

type MaxSimilarityStrategy struct{}

func (s *MaxSimilarityStrategy) Score(segmentEmb []float32, profile types.SpeakerProfile) float32 {
	return BestSimilarity(segmentEmb, profile)
}

type Matcher struct {
	strategy MatchStrategy
}

func NewMatcher(strategy MatchStrategy) *Matcher {
	return &Matcher{strategy: strategy}
}

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
