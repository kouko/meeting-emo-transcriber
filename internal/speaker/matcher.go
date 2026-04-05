package speaker

import (
	"math"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

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

type MatchStrategy interface {
	Score(segmentEmb []float32, profile types.SpeakerProfile) float32
}

type MaxSimilarityStrategy struct{}

func (s *MaxSimilarityStrategy) Score(segmentEmb []float32, profile types.SpeakerProfile) float32 {
	var maxSim float32 = -1
	for _, sample := range profile.Voiceprints {
		sim := CosineSimilarity(segmentEmb, sample.Vector)
		if sim > maxSim {
			maxSim = sim
		}
	}
	return maxSim
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
