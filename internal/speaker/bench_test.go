package speaker

import (
	"math/rand"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// WeSpeaker / FluidAudio embeddings are 256-d float32. The micro-benchmarks
// below use realistic dimensions so the numbers translate to production
// behaviour (a 3-d vector would be 80× faster than reality).
const benchDim = 256

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func randProfile(r *rand.Rand, name string, numVoiceprints int) types.SpeakerProfile {
	vps := make([]types.Voiceprint, numVoiceprints)
	for i := range vps {
		vps[i] = types.Voiceprint{Vector: randVec(r, benchDim)}
	}
	return types.SpeakerProfile{Name: name, Voiceprints: vps}
}

func BenchmarkCosineSimilarity_256d(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	a := randVec(r, benchDim)
	c := randVec(r, benchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = CosineSimilarity(a, c)
	}
}

func BenchmarkBestSimilarity_1Voiceprint(b *testing.B) {
	r := rand.New(rand.NewSource(2))
	emb := randVec(r, benchDim)
	profile := randProfile(r, "Alice", 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BestSimilarity(emb, profile)
	}
}

func BenchmarkBestSimilarity_3Voiceprints(b *testing.B) {
	r := rand.New(rand.NewSource(3))
	emb := randVec(r, benchDim)
	profile := randProfile(r, "Alice", 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = BestSimilarity(emb, profile)
	}
}

func BenchmarkAverageEmbeddings_3Inputs(b *testing.B) {
	r := rand.New(rand.NewSource(4))
	inputs := [][]float32{
		randVec(r, benchDim),
		randVec(r, benchDim),
		randVec(r, benchDim),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = averageEmbeddings(inputs)
	}
}

func BenchmarkAverageEmbeddings_10Inputs(b *testing.B) {
	r := rand.New(rand.NewSource(5))
	inputs := make([][]float32, 10)
	for i := range inputs {
		inputs[i] = randVec(r, benchDim)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = averageEmbeddings(inputs)
	}
}

// ComputeInspection is invoked exactly once by `metr speakers inspect`,
// so per-invocation cost matters more than throughput. The benchmark
// sizes match a typical "5 enrolled speakers, 3 samples per speaker"
// installation.
func BenchmarkComputeInspection_5Speakers(b *testing.B) {
	r := rand.New(rand.NewSource(6))
	target := randVec(r, benchDim)
	fileEmbs := make([][]float32, 3)
	for i := range fileEmbs {
		fileEmbs[i] = randVec(r, benchDim)
	}
	others := make([]types.SpeakerProfile, 5)
	for i := range others {
		others[i] = randProfile(r, "spk"+string(rune('A'+i)), 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ComputeInspection("Alice", fileEmbs, target, others)
	}
}
