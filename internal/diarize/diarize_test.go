package diarize

import (
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestAssignSpeakersExactOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 0.0, End: 5.0, Text: "Hello"},
		{Start: 5.0, End: 10.0, Text: "World"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 5.0, Speaker: 0},
		{Start: 5.0, End: 10.0, Speaker: 1},
	}
	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != 0 {
		t.Errorf("segment 0: got speaker %d, want 0", ids[0])
	}
	if ids[1] != 1 {
		t.Errorf("segment 1: got speaker %d, want 1", ids[1])
	}
}

func TestAssignSpeakersPartialOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 2.0, End: 8.0, Text: "Hello"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 5.0, Speaker: 0},
		{Start: 5.0, End: 10.0, Speaker: 1},
	}
	ids := AssignSpeakers(asrResults, diarSegments)
	// Both overlap 3s — first match wins
	if ids[0] != 0 {
		t.Errorf("got speaker %d, want 0", ids[0])
	}
}

func TestAssignSpeakersNoOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 20.0, End: 25.0, Text: "Late"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 10.0, Speaker: 0},
	}
	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != -1 {
		t.Errorf("got speaker %d, want -1", ids[0])
	}
}

func TestAssignSpeakersEmpty(t *testing.T) {
	ids := AssignSpeakers(nil, nil)
	if len(ids) != 0 {
		t.Errorf("expected empty, got %v", ids)
	}
}
