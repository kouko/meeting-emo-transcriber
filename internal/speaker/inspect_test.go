package speaker

import (
	"math"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestComputeInspection_BasicIntraStats(t *testing.T) {
	target := []float32{1, 0, 0}
	files := [][]float32{
		{0.99, 0.1, 0},  // sim ≈ 0.985
		{0.95, 0.31, 0}, // sim ≈ 0.950
		{0.90, 0.43, 0}, // sim ≈ 0.902
	}
	report := ComputeInspection("Alice", files, target, nil)

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
	report := ComputeInspection("Alice", nil, target, others)

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
	report := ComputeInspection("Alice", nil, target, others)

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
	report := ComputeInspection("Alice", files, target, others)

	wantMargin := report.IntraMin - report.Inter[0].MaxSim
	if math.Abs(float64(report.SafetyMargin-wantMargin)) > 1e-4 {
		t.Errorf("SafetyMargin = %f, want %f", report.SafetyMargin, wantMargin)
	}
}

func TestComputeInspection_NoIntraSamples(t *testing.T) {
	report := ComputeInspection("Alice", nil, []float32{1, 0, 0}, nil)
	if len(report.IntraSims) != 0 {
		t.Errorf("IntraSims len = %d, want 0", len(report.IntraSims))
	}
	if report.IntraMean != 0 || report.IntraMin != 0 || report.IntraMax != 0 {
		t.Errorf("got mean=%f min=%f max=%f, want all 0",
			report.IntraMean, report.IntraMin, report.IntraMax)
	}
}

func TestComputeInspection_NoTargetVoiceprint_StillReturnsReport(t *testing.T) {
	report := ComputeInspection("Alice", [][]float32{{1, 0, 0}}, nil,
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
	report := ComputeInspection("Alice", files, target, nil)
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
