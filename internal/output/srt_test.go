package output

import (
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestSRTFormatter_DCMPFormat(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Start: 0.0, End: 15.2, Speaker: "CEO_Wang", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Great results"},
			{Start: 15.2, End: 30.0, Speaker: "Manager_Lin", Emotion: types.EmotionInfo{Display: ""}, Text: "Here are the numbers"},
		},
	}
	f := &SRTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "(CEO_Wang) [happily] Great results") {
		t.Errorf("expected DCMP format:\n%s", out)
	}
	if !strings.Contains(out, "(Manager_Lin) Here are the numbers") {
		t.Errorf("no emotion tag for Neutral:\n%s", out)
	}
	if !strings.Contains(out, "00:00:00,000 --> 00:00:15,200") {
		t.Errorf("expected SRT timestamp:\n%s", out)
	}
}

func TestFormatSRTTimestamp(t *testing.T) {
	tests := []struct {
		seconds float64
		want    string
	}{
		{0.0, "00:00:00,000"},
		{15.2, "00:00:15,200"},
		{90.5, "00:01:30,500"},
		{3661.123, "01:01:01,123"},
	}
	for _, tt := range tests {
		got := formatSRTTimestamp(tt.seconds)
		if got != tt.want {
			t.Errorf("formatSRTTimestamp(%f) = %q, want %q", tt.seconds, got, tt.want)
		}
	}
}
