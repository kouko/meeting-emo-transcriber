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
		{Start: 0.0, End: 5.0, Speaker: "S1"},
		{Start: 5.0, End: 10.0, Speaker: "S2"},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != "S1" {
		t.Errorf("segment 0: got speaker %q, want S1", ids[0])
	}
	if ids[1] != "S2" {
		t.Errorf("segment 1: got speaker %q, want S2", ids[1])
	}
}

func TestAssignSpeakersPartialOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 2.0, End: 8.0, Text: "Hello"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 5.0, Speaker: "S1"},
		{Start: 5.0, End: 10.0, Speaker: "S2"},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != "S1" {
		t.Errorf("got speaker %q, want S1", ids[0])
	}
}

func TestAssignSpeakersNoOverlap(t *testing.T) {
	asrResults := []types.ASRResult{
		{Start: 20.0, End: 25.0, Text: "Late"},
	}
	diarSegments := []Segment{
		{Start: 0.0, End: 10.0, Speaker: "S1"},
	}

	ids := AssignSpeakers(asrResults, diarSegments)
	if ids[0] != "" {
		t.Errorf("got speaker %q, want empty", ids[0])
	}
}

func TestAssignSpeakersEmpty(t *testing.T) {
	ids := AssignSpeakers(nil, nil)
	if len(ids) != 0 {
		t.Errorf("expected empty, got %v", ids)
	}
}
