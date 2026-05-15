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

// Segment is one contiguous span of audio attributed by diarization to a
// single cluster ID (Speaker). Cluster IDs are stable within a single
// metr-diarize invocation but are NOT enrolled-speaker names — the
// resolver in merge.go is what maps them to names. QualityScore is the
// model's own confidence (0-1); the field is optional because older
// metr-diarize builds did not emit it.
type Segment struct {
	Start        float64 `json:"start"`
	End          float64 `json:"end"`
	Speaker      string  `json:"speaker"`
	QualityScore float64 `json:"quality_score,omitempty"`
}

// DiarizeResult is the parsed JSON output of metr-diarize. Segments is
// time-ordered. SpeakerVoiceprints holds one centroid vector per cluster
// ID (computed by metr-diarize from all segments in that cluster), which
// the matcher uses to compare clusters against enrolled profiles.
type DiarizeResult struct {
	Segments           []Segment            `json:"segments"`
	Speakers           int                  `json:"speakers"`
	SpeakerVoiceprints map[string][]float64 `json:"speaker_voiceprints"`
}

// Process shells out to the metr-diarize Swift CLI to diarize wavPath.
// threshold controls clustering aggressiveness (higher → more speakers);
// numSpeakers > 0 hints the model with an exact count and disables
// auto-detection. The metr-diarize binary itself runs FluidAudio's
// pyannote-segmentation + WeSpeaker + VBx pipeline under the hood and
// returns a single JSON document on stdout.
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

// VoiceprintResult is one entry in metr-diarize's --extract-voiceprints
// output. The Vector is L2-normalised by the underlying WeSpeaker model;
// Model is a string tag (e.g. "fluidaudio_embedding_v1") that callers
// can use to detect when stored vectors were produced by an older
// model and should be invalidated.
type VoiceprintResult struct {
	File   string    `json:"file"`
	Vector []float64 `json:"vector"`
	Dim    int       `json:"dim"`
	Model  string    `json:"model"`
}

// ExtractVoiceprint is the single-file convenience wrapper around
// ExtractVoiceprints. Prefer the batch variant when extracting more than
// one file in a row — model load dominates per-call latency.
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

// ExtractVoiceprints runs metr-diarize once with --extract-voiceprints
// and a batch of wav paths. The model is loaded a single time and reused
// across every input, so callers paying repeated startup cost via
// ExtractVoiceprint (~hundreds of ms each on first-load) should batch
// through here whenever they have more than one file to process.
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
