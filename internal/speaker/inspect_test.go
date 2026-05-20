package speaker

import (
	"math"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// mergedProfileFromVec wraps a single embedding as the "merged" voiceprint
// of a target SpeakerProfile, matching the most common shape these tests
// exercised before ComputeInspection accepted full profiles.
func mergedProfileFromVec(v []float32) types.SpeakerProfile {
	if len(v) == 0 {
		return types.SpeakerProfile{}
	}
	return types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{{Type: "merged", Vector: v}},
	}
}

func TestComputeInspection_BasicIntraStats(t *testing.T) {
	target := []float32{1, 0, 0}
	files := [][]float32{
		{0.99, 0.1, 0},  // sim ≈ 0.985
		{0.95, 0.31, 0}, // sim ≈ 0.950
		{0.90, 0.43, 0}, // sim ≈ 0.902
	}
	report := ComputeInspection("Alice", files, mergedProfileFromVec(target), nil)

	if len(report.IntraSims) != 3 {
		t.Fatalf("IntraSims len = %d, want 3", len(report.IntraSims))
	}
	if math.Abs(float64(report.IntraMax-report.IntraSims[0])) > 1e-4 {
		t.Errorf("Max = %f, want %f", report.IntraMax, report.IntraSims[0])
	}
	if math.Abs(float64(report.IntraMin-report.IntraSims[2])) > 1e-4 {
		t.Errorf("Min = %f, want %f", report.IntraMin, report.IntraSims[2])
	}
	wantMean := (report.IntraSims[0] + report.IntraSims[1] + report.IntraSims[2]) / 3
	if math.Abs(float64(report.IntraMean-wantMean)) > 1e-4 {
		t.Errorf("Mean = %f, want %f", report.IntraMean, wantMean)
	}
}

func TestComputeInspection_InterSortedDescending(t *testing.T) {
	target := []float32{1, 0, 0}
	others := []types.SpeakerProfile{
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 1, 0}}}},     // sim 0
		{Name: "Carol", Voiceprints: []types.Voiceprint{{Vector: []float32{0.8, 0.6, 0}}}}, // sim 0.8
		{Name: "Dan", Voiceprints: []types.Voiceprint{{Vector: []float32{0.5, 0.866, 0}}}}, // sim 0.5
	}
	report := ComputeInspection("Alice", nil, mergedProfileFromVec(target), others)

	if len(report.Inter) != 3 {
		t.Fatalf("Inter len = %d, want 3", len(report.Inter))
	}
	if report.Inter[0].OtherName != "Carol" {
		t.Errorf("strongest impostor = %q, want Carol", report.Inter[0].OtherName)
	}
	if report.Inter[2].OtherName != "Bob" {
		t.Errorf("weakest impostor = %q, want Bob", report.Inter[2].OtherName)
	}
}

func TestComputeInspection_SelfNotInIntra(t *testing.T) {
	// The target speaker's profile is provided in otherProfiles; the
	// function should filter itself out.
	target := []float32{1, 0, 0}
	others := []types.SpeakerProfile{
		{Name: "Alice", Voiceprints: []types.Voiceprint{{Vector: []float32{1, 0, 0}}}}, // self
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 1, 0}}}},
	}
	report := ComputeInspection("Alice", nil, mergedProfileFromVec(target), others)

	if len(report.Inter) != 1 {
		t.Fatalf("Inter len = %d, want 1 (self filtered)", len(report.Inter))
	}
	if report.Inter[0].OtherName != "Bob" {
		t.Errorf("got %q, want Bob", report.Inter[0].OtherName)
	}
}

func TestComputeInspection_SafetyMargin(t *testing.T) {
	target := []float32{1, 0, 0}
	files := [][]float32{
		{0.99, 0.1, 0},
		{0.95, 0.31, 0},
	}
	others := []types.SpeakerProfile{
		// Bob is close to Alice — sim 0.8 — limits separation.
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0.8, 0.6, 0}}}},
	}
	report := ComputeInspection("Alice", files, mergedProfileFromVec(target), others)

	wantMargin := report.IntraMin - report.Inter[0].MaxSim
	if math.Abs(float64(report.SafetyMargin-wantMargin)) > 1e-4 {
		t.Errorf("SafetyMargin = %f, want %f", report.SafetyMargin, wantMargin)
	}
}

func TestComputeInspection_NoIntraSamples(t *testing.T) {
	report := ComputeInspection("Alice", nil, mergedProfileFromVec([]float32{1, 0, 0}), nil)
	if len(report.IntraSims) != 0 {
		t.Errorf("IntraSims len = %d, want 0", len(report.IntraSims))
	}
	if report.IntraMean != 0 || report.IntraMin != 0 || report.IntraMax != 0 {
		t.Errorf("got mean=%f min=%f max=%f, want all 0",
			report.IntraMean, report.IntraMin, report.IntraMax)
	}
}

func TestComputeInspection_NoTargetVoiceprint_StillReturnsReport(t *testing.T) {
	report := ComputeInspection("Alice", [][]float32{{1, 0, 0}}, types.SpeakerProfile{},
		[]types.SpeakerProfile{
			{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0, 1, 0}}}},
		})
	if len(report.IntraSims) != 0 {
		t.Errorf("IntraSims should be empty when target voiceprint missing")
	}
	if len(report.Inter) != 0 {
		t.Errorf("Inter should be empty when target voiceprint missing")
	}
}

func TestComputeInspection_EmptyEmbeddingsSkipped(t *testing.T) {
	target := []float32{1, 0, 0}
	files := [][]float32{
		{},              // empty — should be skipped
		{0.99, 0.1, 0},
	}
	report := ComputeInspection("Alice", files, mergedProfileFromVec(target), nil)
	if len(report.IntraSims) != 1 {
		t.Errorf("IntraSims len = %d, want 1 (empty skipped)", len(report.IntraSims))
	}
}

func TestMergedVoiceprint_PicksMergedType(t *testing.T) {
	profile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{1, 0, 0}},
			{Type: "merged", Vector: []float32{0, 1, 0}},
		},
	}
	got := MergedVoiceprint(&profile)
	if len(got) == 0 || got[1] != 1 {
		t.Errorf("got %v, want the merged voiceprint", got)
	}
}

func TestMergedVoiceprint_NoMergedType_ReturnsNil(t *testing.T) {
	profile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{1, 0, 0}},
		},
	}
	if got := MergedVoiceprint(&profile); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

func TestMergedVoiceprint_NilProfile(t *testing.T) {
	if got := MergedVoiceprint(nil); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

// TestComputeInspection_BestOfAllMargin_UsesMaxOverProfile constructs a
// pathological enrollment: the "merged" voiceprint is a stale one that
// barely correlates with the per-file embeddings, while a second
// "centroid" voiceprint in the same profile correlates strongly.
// The merged-based margin should report a tight / negative number, but
// the best-of-all margin (matching live matcher behaviour) should report
// a comfortable positive margin.
func TestComputeInspection_BestOfAllMargin_UsesMaxOverProfile(t *testing.T) {
	files := [][]float32{
		{1, 0, 0},
		{0.99, 0.14, 0},
		{0.97, 0.24, 0},
	}
	targetProfile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{1, 0, 0}}, // matches files ~1.0
			{Type: "merged", Vector: []float32{0, 1, 0}},   // matches files ~0.0
		},
	}
	others := []types.SpeakerProfile{
		{Name: "Bob", Voiceprints: []types.Voiceprint{{Vector: []float32{0.3, 0.95, 0}}}}, // sim ~0.3 vs files
	}

	report := ComputeInspection("Alice", files, targetProfile, others)

	// Merged-based intra-min is near 0 (files vs {0,1,0}).
	if report.IntraMin > 0.2 {
		t.Errorf("merged IntraMin = %.3f, want near 0 (files barely match stale merged)", report.IntraMin)
	}
	// Best-of-all intra-min is near 1.0 (files match the centroid voiceprint).
	if report.BestOfAllIntraMin < 0.9 {
		t.Errorf("BestOfAllIntraMin = %.3f, want >= 0.9 (files match centroid voiceprint)", report.BestOfAllIntraMin)
	}
	// Merged-margin is negative (stale merged loses to impostor).
	if report.SafetyMargin >= 0 {
		t.Errorf("merged SafetyMargin = %.3f, want < 0 (stale merged loses to impostor)", report.SafetyMargin)
	}
	// Best-of-all margin is comfortably positive — that's the whole point.
	if report.BestOfAllSafetyMargin <= 0 {
		t.Errorf("BestOfAllSafetyMargin = %.3f, want > 0 (centroid voiceprint rescues the speaker)", report.BestOfAllSafetyMargin)
	}
	// The two margins should differ by a wide gap — the merged stat is
	// pathologically misleading here while best-of-all is realistic.
	gap := report.BestOfAllSafetyMargin - report.SafetyMargin
	if gap < 0.5 {
		t.Errorf("margin gap = %.3f, want >= 0.5 (merged vs best-of-all should diverge sharply)", gap)
	}
}
