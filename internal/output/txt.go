package output

import (
	"fmt"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type TXTFormatter struct{}

func (f *TXTFormatter) Format(result types.TranscriptResult) (string, error) {
	var b strings.Builder
	var currentSpeaker string
	var currentEmotion string

	for _, seg := range result.Segments {
		eventDisplay := types.AudioEventDisplayMap[seg.AudioEvent]
		if eventDisplay != "" && seg.Text == "" {
			if currentSpeaker != "" {
				fmt.Fprintf(&b, "%s\n", eventDisplay)
			} else {
				fmt.Fprintf(&b, "%s\n", eventDisplay)
			}
			continue
		}

		speakerChanged := seg.Speaker != currentSpeaker
		if speakerChanged {
			if currentSpeaker != "" {
				b.WriteString("\n")
			}
			currentSpeaker = seg.Speaker
			currentEmotion = ""
			if seg.Emotion.Display != "" {
				fmt.Fprintf(&b, "%s [%s]\n", seg.Speaker, seg.Emotion.Display)
				currentEmotion = seg.Emotion.Display
			} else {
				fmt.Fprintf(&b, "%s\n", seg.Speaker)
			}
		} else if seg.Emotion.Display != currentEmotion && seg.Emotion.Display != "" {
			fmt.Fprintf(&b, "[%s] ", seg.Emotion.Display)
			currentEmotion = seg.Emotion.Display
		}

		if eventDisplay != "" {
			fmt.Fprintf(&b, "%s ", eventDisplay)
		}
		if seg.Text != "" {
			fmt.Fprintf(&b, "%s\n", seg.Text)
		}
	}
	return b.String(), nil
}
