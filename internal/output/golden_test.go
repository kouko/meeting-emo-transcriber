package output

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// updateGolden regenerates testdata/*.golden files instead of asserting
// against them. Run `go test ./internal/output/ -update` after intentional
// format changes, then review the diff before committing.
var updateGolden = flag.Bool("update", false, "regenerate golden output snapshots")

// checkGolden compares got against the contents of testdata/<name>.golden,
// regenerating the file when -update is set.
func checkGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name+".golden")
	if *updateGolden {
		if err := os.WriteFile(path, []byte(got), 0644); err != nil {
			t.Fatalf("write golden %s: %v", path, err)
		}
		t.Logf("updated %s", path)
		return
	}
	wantBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s (run with -update to create): %v", path, err)
	}
	want := string(wantBytes)
	if got != want {
		t.Errorf("%s snapshot mismatch\n--- got (%d bytes) ---\n%s\n--- want (%d bytes) ---\n%s",
			name, len(got), got, len(want), want)
		// Also show first diverging line for easier eyeballing.
		gotLines := strings.SplitAfter(got, "\n")
		wantLines := strings.SplitAfter(want, "\n")
		for i := 0; i < len(gotLines) || i < len(wantLines); i++ {
			var gl, wl string
			if i < len(gotLines) {
				gl = gotLines[i]
			}
			if i < len(wantLines) {
				wl = wantLines[i]
			}
			if gl != wl {
				t.Logf("first diff at line %d:\n  got:  %q\n  want: %q", i+1, gl, wl)
				break
			}
		}
	}
}

// --- shared fixtures -------------------------------------------------------

func meetingTranscript() types.TranscriptResult {
	return types.TranscriptResult{
		Metadata: types.Metadata{
			File:               "meeting.mp3",
			Duration:           "00:05:30",
			SpeakersDetected:   3,
			SpeakersIdentified: 2,
			Date:               "2024-01-15T10:30:00Z",
		},
		Segments: []types.TranscriptSegment{
			{
				Start:      0.0,
				End:        4.5,
				Speaker:    "Alice",
				Emotion:    types.EmotionInfo{Raw: "HAPPY", Label: "Happy", Display: "happily"},
				AudioEvent: "Speech",
				Language:   "zh",
				Text:       "大家早安",
			},
			{
				Start:      4.5,
				End:        7.2,
				Speaker:    "Alice",
				Emotion:    types.EmotionInfo{Raw: "HAPPY", Label: "Happy", Display: "happily"},
				AudioEvent: "Speech",
				Language:   "zh",
				Text:       "今天會議的重點有三個",
			},
			{
				Start:      7.5,
				End:        9.0,
				Speaker:    "Alice",
				Emotion:    types.EmotionInfo{Raw: "NEUTRAL", Label: "Neutral", Display: ""},
				AudioEvent: "Laughter",
				Language:   "zh",
				Text:       "",
			},
			{
				Start:      9.5,
				End:        14.0,
				Speaker:    "Bob",
				Emotion:    types.EmotionInfo{Raw: "ANGRY", Label: "Angry", Display: "angrily"},
				AudioEvent: "Speech",
				Language:   "zh",
				Text:       "我有不同意見",
			},
			{
				Start:      14.0,
				End:        18.0,
				Speaker:    "Unknown",
				Emotion:    types.EmotionInfo{Raw: "NEUTRAL", Label: "Neutral", Display: ""},
				AudioEvent: "Speech",
				Language:   "en",
				Text:       "Excuse me, can I add something?",
			},
		},
	}
}

// --- snapshot tests --------------------------------------------------------

func TestJSONFormatter_Golden(t *testing.T) {
	out, err := (&JSONFormatter{}).Format(meetingTranscript())
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "transcript.json", out)
}

func TestSRTFormatter_Golden(t *testing.T) {
	out, err := (&SRTFormatter{}).Format(meetingTranscript())
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "transcript.srt", out)
}

func TestTXTFormatter_Golden(t *testing.T) {
	out, err := (&TXTFormatter{}).Format(meetingTranscript())
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "transcript.txt", out)
}

func TestTXTFormatter_WithPunctuation_Golden(t *testing.T) {
	// Fake punctuator appends "。" if Chinese, "." otherwise.
	punct := func(s string) string {
		if strings.ContainsAny(s, "你我他大家今天會議") {
			return s + "。"
		}
		return s + "."
	}
	f := &TXTFormatter{PunctFunc: punct}
	out, err := f.Format(meetingTranscript())
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "transcript.punctuated.txt", out)
}

// --- edge case snapshots ---------------------------------------------------

func TestSRTFormatter_EmptyTranscript_Golden(t *testing.T) {
	out, err := (&SRTFormatter{}).Format(types.TranscriptResult{})
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "empty.srt", out)
}

func TestTXTFormatter_StandaloneAudioEvent_Golden(t *testing.T) {
	// Audio event at the very start, with no preceding speaker — exercises
	// the "standalone event before any speaker" branch in txt.go:36.
	result := types.TranscriptResult{
		Segments: []types.TranscriptSegment{
			{Start: 0.0, End: 2.0, Speaker: "Unknown", AudioEvent: "Applause", Text: ""},
			{Start: 2.0, End: 6.0, Speaker: "Alice", AudioEvent: "Speech", Text: "謝謝大家"},
		},
	}
	out, err := (&TXTFormatter{}).Format(result)
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "standalone-event.txt", out)
}
