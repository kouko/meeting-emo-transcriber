// Package speaker manages the on-disk speaker store and the cosine-based
// matching logic that maps diarization clusters to enrolled names.
package speaker

import (
	"math"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// CosineSimilarity returns the cosine of the angle between a and b. The
// chosen score range is [-1, 1] with 1 meaning identical direction.
//
// Returns 0 (rather than panicking) when inputs have different lengths,
// when either is empty, or when either has zero norm — these are defensive
// guards that fire when a stored voiceprint was produced by a different
// model than the query, or when the audio segment was pure silence.
// Callers that need to distinguish "zero similarity" from "incompatible
// input" should validate dimensions themselves before calling.
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

// MatchStrategy decides how to score a single embedding against a profile
// that may contain several stored voiceprints. The abstraction exists so
// future strategies (e.g. attention-pooled enrollment, AS-Norm) can be
// dropped in without touching call sites — see config.Strategy for the
// selector wired into the CLI.
type MatchStrategy interface {
	Score(segmentEmb []float32, profile types.SpeakerProfile) float32
}

// MaxSimilarityStrategy scores a profile by its highest-similarity stored
// voiceprint. This is the only strategy currently implemented and is the
// default used by both the standalone matcher (cmd/commands/speakers.go)
// and the cluster resolver in internal/diarize.
type MaxSimilarityStrategy struct{}

// Score delegates to BestSimilarity — see that function's docstring.
func (s *MaxSimilarityStrategy) Score(segmentEmb []float32, profile types.SpeakerProfile) float32 {
	return BestSimilarity(segmentEmb, profile)
}

// Matcher is the single-shot "given one embedding, which enrolled speaker
// (if any) does it belong to?" entrypoint. Unlike the cluster resolver in
// internal/diarize, this path does not enforce one-to-one assignment or a
// margin guard — it is used by `metr speakers verify --audio` where the
// caller already knows there is exactly one input embedding.
type Matcher struct {
	strategy MatchStrategy
}

// NewMatcher returns a Matcher backed by the given scoring strategy.
func NewMatcher(strategy MatchStrategy) *Matcher {
	return &Matcher{strategy: strategy}
}

// Match scores embedding against every profile and returns the closest
// one, or an empty Name when no profile clears threshold. Similarity is
// always populated with the highest score seen so callers can show "best
// guess was X with sim 0.42" even on rejection.
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
