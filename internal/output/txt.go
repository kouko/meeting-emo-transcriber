package output

import (
	"fmt"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// TXTFormatter formats transcript as readable text, merging consecutive segments
// from the same speaker with the same emotion into paragraphs.
// If PunctFunc is set, punctuation is applied to each merged paragraph.
type TXTFormatter struct {
	PunctFunc func(string) string
}

// paragraph holds accumulated text lines for a speaker+emotion group.
type paragraph struct {
	Speaker  string
	Emotion  string
	Lines    []string
	Events   []string // audio event displays interspersed
}

func (f *TXTFormatter) Format(result types.TranscriptResult) (string, error) {
	// First pass: group segments into paragraphs by speaker+emotion
	var paragraphs []paragraph
	var current *paragraph

	for _, seg := range result.Segments {
		eventDisplay := types.AudioEventDisplayMap[seg.AudioEvent]
		if eventDisplay != "" && seg.Text == "" {
			if current != nil {
				current.Events = append(current.Events, eventDisplay)
			} else {
				// Standalone event before any speaker
				paragraphs = append(paragraphs, paragraph{
					Speaker: seg.Speaker,
					Events:  []string{eventDisplay},
				})
			}
			continue
		}

		speakerChanged := current == nil || seg.Speaker != current.Speaker
		emotionChanged := current != nil && seg.Emotion.Display != current.Emotion && seg.Emotion.Display != ""
		if speakerChanged || emotionChanged {
			if current != nil {
				paragraphs = append(paragraphs, *current)
			}
			current = &paragraph{
				Speaker: seg.Speaker,
				Emotion: seg.Emotion.Display,
			}
		}

		line := ""
		if eventDisplay != "" {
			line += eventDisplay + " "
		}
		if seg.Text != "" {
			line += seg.Text
		}
		if line != "" {
			current.Lines = append(current.Lines, line)
		}
	}
	if current != nil {
		paragraphs = append(paragraphs, *current)
	}

	// Second pass: render paragraphs, applying punctuation to merged text
	var b strings.Builder
	for i, p := range paragraphs {
		if i > 0 {
			b.WriteString("\n")
		}
		if p.Emotion != "" {
			fmt.Fprintf(&b, "%s [%s]\n", p.Speaker, p.Emotion)
		} else {
			fmt.Fprintf(&b, "%s\n", p.Speaker)
		}

		// Merge all lines into one text block for punctuation
		merged := strings.Join(p.Lines, " ")
		if f.PunctFunc != nil && merged != "" {
			merged = f.PunctFunc(merged)
		}
		if merged != "" {
			fmt.Fprintf(&b, "%s\n", merged)
		}

		for _, ev := range p.Events {
			fmt.Fprintf(&b, "%s\n", ev)
		}
	}
	return b.String(), nil
}
