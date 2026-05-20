package diarize

import (
	"math/rand"
	"os"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// silenceStderr redirects os.Stderr for the duration of t, restoring it on
// cleanup. Used so per-iteration diagnostic prints don't garble the
// benchmark output line.
func silenceStderr(b *testing.B) {
	b.Helper()
	orig := os.Stderr
	null, err := os.Open(os.DevNull)
	if err != nil {
		b.Fatal(err)
	}
	os.Stderr = null
	b.Cleanup(func() {
		os.Stderr = orig
		null.Close()
	})
}

const benchDim = 256

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func randProfile(r *rand.Rand, name string, n int) types.SpeakerProfile {
	vps := make([]types.Voiceprint, n)
	for i := range vps {
		vps[i] = types.Voiceprint{Vector: randVec(r, benchDim)}
	}
	return types.SpeakerProfile{Name: name, Voiceprints: vps}
}

// matchAgainstProfilesDetailed is called once per cluster during
// ResolveSpeakerNames. The benchmarks below model realistic meeting
// sizes (3 to 10 clusters × 1-2 voiceprints per enrolled speaker).
func BenchmarkMatchAgainstProfilesDetailed_5Profiles(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	emb := randVec(r, benchDim)
	profiles := make([]types.SpeakerProfile, 5)
	for i := range profiles {
		profiles[i] = randProfile(r, "spk"+string(rune('A'+i)), 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = matchAgainstProfilesDetailed(emb, profiles, 0.65)
	}
}

func BenchmarkMatchAgainstProfilesDetailed_20Profiles_2VoiceprintsEach(b *testing.B) {
	r := rand.New(rand.NewSource(2))
	emb := randVec(r, benchDim)
	profiles := make([]types.SpeakerProfile, 20)
	for i := range profiles {
		profiles[i] = randProfile(r, "spk"+string(rune('A'+i%26)), 2)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = matchAgainstProfilesDetailed(emb, profiles, 0.65)
	}
}

// RefineSpeakerNamesPerSegment with a no-op extractor measures the Go
// bookkeeping cost — wav extraction + writing dominates in production
// but this isolates the policy logic.
func BenchmarkRefineSpeakerNamesPerSegment_50Segments(b *testing.B) {
	r := rand.New(rand.NewSource(3))
	const n = 50
	names := make([]string, n)
	asr := make([]types.ASRResult, n)
	for i := range names {
		names[i] = "Alice"
		asr[i] = types.ASRResult{Start: float64(i) * 2, End: float64(i)*2 + 1.5, Text: "hi"}
	}
	profiles := []types.SpeakerProfile{randProfile(r, "Alice", 1)}
	// 30 seconds of fake samples at 16 kHz.
	wavSamples := make([]float32, 16000*100)
	for i := range wavSamples {
		wavSamples[i] = r.Float32()*0.1 - 0.05
	}

	// Pre-build the embeddings the fake extractor will return.
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = randVec(r, benchDim)
	}
	batchExtract := func(paths []string) ([][]float32, error) {
		// Return enough embeddings for the call.
		return embs[:len(paths)], nil
	}

	tmpDir := b.TempDir()
	silenceStderr(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = RefineSpeakerNamesPerSegment(names, asr, profiles, wavSamples, 16000,
			batchExtract, 0.5, 1.0, tmpDir)
	}
}
