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

// DiarizeResult contains diarization segments and per-speaker embeddings.
type DiarizeResult struct {
	Segments          []Segment              `json:"segments"`
	Speakers          int                    `json:"speakers"`
	SpeakerEmbeddings map[string][]float64   `json:"speaker_embeddings"`
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

// EmbeddingResult is the JSON output from metr-diarize --extract-embedding.
type EmbeddingResult struct {
	Embedding []float64 `json:"embedding"`
	Dim       int       `json:"dim"`
	Model     string    `json:"model"`
}

// ExtractEmbedding runs metr-diarize in embedding extraction mode.
// Returns a single speaker embedding for the given audio file.
func ExtractEmbedding(binPath, wavPath string) (*EmbeddingResult, error) {
	cmd := exec.Command(binPath, "--extract-embedding", wavPath)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("metr-diarize extract-embedding failed: %w", err)
	}

	var result EmbeddingResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return nil, fmt.Errorf("parse embedding output: %w", err)
	}

	return &result, nil
}
