package output

import (
	"encoding/json"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type JSONFormatter struct{}

func (f *JSONFormatter) Format(result types.TranscriptResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data) + "\n", nil
}
