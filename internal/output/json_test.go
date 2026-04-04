package output

import (
	"encoding/json"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestJSONFormatter_ValidJSON(t *testing.T) {
	result := types.TranscriptResult{
		Metadata: types.Metadata{File: "test.wav", Duration: "00:01:00"},
		Segments: []types.TranscriptSegment{
			{Start: 0.0, End: 5.0, Speaker: "Alice", Emotion: types.EmotionInfo{Raw: "HAPPY", Label: "Happy", Display: "happily"}, Text: "Hello"},
		},
	}
	f := &JSONFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	var parsed types.TranscriptResult
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out)
	}
	if parsed.Segments[0].Emotion.Raw != "HAPPY" {
		t.Errorf("emotion.raw = %q, want HAPPY", parsed.Segments[0].Emotion.Raw)
	}
}
