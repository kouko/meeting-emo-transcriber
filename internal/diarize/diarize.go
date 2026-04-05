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

// DiarizeResult contains diarization segments and per-speaker voiceprints.
type DiarizeResult struct {
	Segments           []Segment            `json:"segments"`
	Speakers           int                  `json:"speakers"`
	SpeakerVoiceprints map[string][]float64 `json:"speaker_voiceprints"`
}

// Process runs speaker diarization on a WAV file using metr-diarize CLI.
// Returns diarization result including segments and per-speaker centroid embeddings.
func Process(binPath, wavPath string, threshold float32, numSpeakers int) (*DiarizeResult, error) {
	args := []string{wavPath, "--threshold", strconv.FormatFloat(float64(threshold), 'f', 2, 32)}
	if numSpeakers > 0 {
		args = append(args, "--num-speakers", strconv.Itoa(numSpeakers))
	}

	cmd := exec.Command(binPath, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("metr-diarize failed: %w", err)
	}

	var result DiarizeResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("parse diarize output: %w", err)
	}

	return &result, nil
}

// VoiceprintResult is the JSON output from metr-diarize --extract-voiceprints.
type VoiceprintResult struct {
	File   string    `json:"file"`
	Vector []float64 `json:"vector"`
	Dim    int       `json:"dim"`
	Model  string    `json:"model"`
}

// ExtractVoiceprint runs metr-diarize in voiceprint extraction mode for a single file.
func ExtractVoiceprint(binPath, wavPath string) (*VoiceprintResult, error) {
	results, err := ExtractVoiceprints(binPath, []string{wavPath})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("no voiceprint extracted")
	}
	return &results[0], nil
}

// ExtractVoiceprints runs metr-diarize in batch voiceprint extraction mode.
// Loads the model once for all files. Much faster than calling ExtractVoiceprint per file.
func ExtractVoiceprints(binPath string, wavPaths []string) ([]VoiceprintResult, error) {
	args := append([]string{"--extract-voiceprints"}, wavPaths...)
	cmd := exec.Command(binPath, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("metr-diarize extract-voiceprints failed: %w", err)
	}

	var results []VoiceprintResult
	if err := json.Unmarshal(stdout.Bytes(), &results); err != nil {
		return nil, fmt.Errorf("parse voiceprints output: %w", err)
	}

	return results, nil
}
