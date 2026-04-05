// Package diarize provides speaker diarization via the metr-diarize CLI tool
// (FluidAudio-based, CoreML/ANE accelerated).
package diarize

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strconv"
)

// Segment represents a speaker diarization result.
type Segment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}

// diarizeOutput is the JSON format from metr-diarize CLI.
type diarizeOutput struct {
	Segments []Segment `json:"segments"`
	Speakers int       `json:"speakers"`
}

// Process runs speaker diarization on a WAV file using metr-diarize CLI.
// binPath: path to metr-diarize binary
// wavPath: path to 16kHz mono WAV file
// threshold: clustering threshold (higher = fewer speakers)
// numSpeakers: if > 0, fixed speaker count; otherwise auto-detect
// Returns segments sorted by start time.
func Process(binPath, wavPath string, threshold float32, numSpeakers int) ([]Segment, error) {
	args := []string{wavPath, "--threshold", strconv.FormatFloat(float64(threshold), 'f', 2, 32)}
	if numSpeakers > 0 {
		args = append(args, "--num-speakers", strconv.Itoa(numSpeakers))
	}

	cmd := exec.Command(binPath, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr // show progress logs to user

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("metr-diarize failed: %w", err)
	}

	var output diarizeOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		return nil, fmt.Errorf("parse diarize output: %w", err)
	}

	return output.Segments, nil
}
