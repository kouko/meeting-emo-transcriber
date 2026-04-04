package output

import (
	"fmt"
	"math"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type SRTFormatter struct{}

func (f *SRTFormatter) Format(result types.TranscriptResult) (string, error) {
	var b strings.Builder
	for i, seg := range result.Segments {
		fmt.Fprintf(&b, "%d\n", i+1)
		fmt.Fprintf(&b, "%s --> %s\n", formatSRTTimestamp(seg.Start), formatSRTTimestamp(seg.End))
		var line strings.Builder
		fmt.Fprintf(&line, "(%s)", seg.Speaker)
		if seg.Emotion.Display != "" {
			fmt.Fprintf(&line, " [%s]", seg.Emotion.Display)
		}
		fmt.Fprintf(&line, " %s", seg.Text)
		fmt.Fprintf(&b, "%s\n\n", strings.TrimSpace(line.String()))
	}
	return b.String(), nil
}

func formatSRTTimestamp(seconds float64) string {
	totalMs := int(math.Round(seconds * 1000))
	h := totalMs / 3600000
	totalMs %= 3600000
	m := totalMs / 60000
	totalMs %= 60000
	s := totalMs / 1000
	ms := totalMs % 1000
	return fmt.Sprintf("%02d:%02d:%02d,%03d", h, m, s, ms)
}
