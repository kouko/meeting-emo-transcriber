package asr

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// ParseSRTTimestamp parses "HH:MM:SS,mmm" into seconds as float64.
func ParseSRTTimestamp(ts string) (float64, error) {
	parts := strings.Split(ts, ":")
	if len(parts) != 3 {
		return 0, fmt.Errorf("invalid SRT timestamp: %q", ts)
	}

	h, err := strconv.Atoi(parts[0])
	if err != nil || len(parts[0]) != 2 {
		return 0, fmt.Errorf("invalid hours in SRT timestamp: %q", ts)
	}

	m, err := strconv.Atoi(parts[1])
	if err != nil || len(parts[1]) != 2 {
		return 0, fmt.Errorf("invalid minutes in SRT timestamp: %q", ts)
	}

	secParts := strings.Split(parts[2], ",")
	if len(secParts) != 2 || len(secParts[0]) != 2 || len(secParts[1]) != 3 {
		return 0, fmt.Errorf("invalid seconds in SRT timestamp: %q", ts)
	}

	s, err := strconv.Atoi(secParts[0])
	if err != nil {
		return 0, fmt.Errorf("invalid seconds in SRT timestamp: %q", ts)
	}

	ms, err := strconv.Atoi(secParts[1])
	if err != nil {
		return 0, fmt.Errorf("invalid milliseconds in SRT timestamp: %q", ts)
	}

	return float64(h)*3600 + float64(m)*60 + float64(s) + float64(ms)/1000.0, nil
}

// ParseSRT parses SRT-formatted text into ASRResult slices.
// language is passed through to each result (whisper SRT has no language metadata).
func ParseSRT(content string, language string) ([]types.ASRResult, error) {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil, nil
	}

	content = strings.ReplaceAll(content, "\r\n", "\n")
	blocks := strings.Split(content, "\n\n")

	var results []types.ASRResult
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		lines := strings.Split(block, "\n")
		if len(lines) < 2 {
			continue
		}

		// Line 0: sequence number (skip)
		// Line 1: "HH:MM:SS,mmm --> HH:MM:SS,mmm"
		// Lines 2+: text
		tsLine := strings.TrimSpace(lines[1])
		tsParts := strings.Split(tsLine, " --> ")
		if len(tsParts) != 2 {
			continue
		}

		start, err := ParseSRTTimestamp(strings.TrimSpace(tsParts[0]))
		if err != nil {
			return nil, fmt.Errorf("parse start timestamp: %w", err)
		}

		end, err := ParseSRTTimestamp(strings.TrimSpace(tsParts[1]))
		if err != nil {
			return nil, fmt.Errorf("parse end timestamp: %w", err)
		}

		var textParts []string
		for _, line := range lines[2:] {
			line = strings.TrimSpace(line)
			if line != "" {
				textParts = append(textParts, line)
			}
		}
		text := strings.Join(textParts, " ")
		if text == "" {
			continue
		}

		results = append(results, types.ASRResult{
			Start:    start,
			End:      end,
			Text:     text,
			Language: language,
		})
	}

	return results, nil
}
