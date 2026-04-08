package diarize

import (
	"encoding/json"
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

func TestMaxSegmentDuration(t *testing.T) {
	segments := []Segment{
		{Start: 0, End: 3, Speaker: "S1"},
		{Start: 5, End: 25, Speaker: "S1"},
		{Start: 30, End: 35, Speaker: "S2"},
	}

	if got := maxSegmentDuration(segments, "S1"); got != 20.0 {
		t.Errorf("S1 max duration: got %.1f, want 20.0", got)
	}
	if got := maxSegmentDuration(segments, "S2"); got != 5.0 {
		t.Errorf("S2 max duration: got %.1f, want 5.0", got)
	}
	if got := maxSegmentDuration(segments, "S3"); got != 0.0 {
		t.Errorf("S3 max duration: got %.1f, want 0.0", got)
	}
}

func TestMaxSegmentDurationEmpty(t *testing.T) {
	if got := maxSegmentDuration(nil, "S1"); got != 0.0 {
		t.Errorf("nil segments: got %.1f, want 0.0", got)
	}
}

func TestSegmentQualityScoreJSON(t *testing.T) {
	// JSON with quality_score
	data := `{"start":1.0,"end":5.0,"speaker":"S1","quality_score":0.85}`
	var seg Segment
	if err := json.Unmarshal([]byte(data), &seg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if seg.QualityScore != 0.85 {
		t.Errorf("QualityScore = %f, want 0.85", seg.QualityScore)
	}

	// JSON without quality_score (backward compat)
	data2 := `{"start":1.0,"end":5.0,"speaker":"S1"}`
	var seg2 Segment
	if err := json.Unmarshal([]byte(data2), &seg2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if seg2.QualityScore != 0.0 {
		t.Errorf("QualityScore = %f, want 0.0 (zero value)", seg2.QualityScore)
	}
}

func TestDiarizeResultQualityScoreJSON(t *testing.T) {
	data := `{
		"segments": [
			{"start":0,"end":10,"speaker":"S1","quality_score":0.9},
			{"start":10,"end":20,"speaker":"S2"}
		],
		"speakers": 2,
		"speaker_voiceprints": {}
	}`
	var result DiarizeResult
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(result.Segments) != 2 {
		t.Fatalf("segments count = %d, want 2", len(result.Segments))
	}
	if result.Segments[0].QualityScore != 0.9 {
		t.Errorf("seg[0].QualityScore = %f, want 0.9", result.Segments[0].QualityScore)
	}
	if result.Segments[1].QualityScore != 0.0 {
		t.Errorf("seg[1].QualityScore = %f, want 0.0", result.Segments[1].QualityScore)
	}
}
