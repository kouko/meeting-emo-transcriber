package output

import (
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestTXTFormatter_BasicOutput(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Hello everyone"},
			{Speaker: "Bob", Emotion: types.EmotionInfo{Display: ""}, Text: "Hi Alice"},
		},
	}
	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Alice [happily]") {
		t.Error("expected 'Alice [happily]'")
	}
	if !strings.Contains(out, "Hello everyone") {
		t.Error("expected text")
	}
	if strings.Contains(out, "Bob [") {
		t.Error("Bob should not have emotion tag")
	}
	if !strings.Contains(out, "Bob\n") {
		t.Error("expected 'Bob' on its own line")
	}
}

func TestTXTFormatter_MergeSameSpeaker(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "First line"},
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Second line"},
			{Speaker: "Bob", Emotion: types.EmotionInfo{Display: ""}, Text: "Bob speaks"},
		},
	}
	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(out, "Alice") != 1 {
		t.Errorf("Alice should appear once:\n%s", out)
	}
	if !strings.Contains(out, "First line Second line") {
		t.Errorf("lines should be merged into paragraph:\n%s", out)
	}
}

func TestTXTFormatter_EmotionChange(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "happily"}, Text: "Good news"},
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: "angrily"}, Text: "But this is bad"},
		},
	}
	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "[angrily]") {
		t.Errorf("expected emotion change:\n%s", out)
	}
}

func TestTXTFormatter_AudioEvent(t *testing.T) {
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Speaker: "Alice", Emotion: types.EmotionInfo{Display: ""}, AudioEvent: "Laughter", Text: ""},
		},
	}
	f := &TXTFormatter{}
	out, err := f.Format(result)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "[laughter]") {
		t.Errorf("expected audio event:\n%s", out)
	}
}
