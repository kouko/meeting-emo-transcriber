// Package emotion classifies speech emotion and detects audio events
// using the SenseVoice model. The model is not loaded in-process; this
// package is a thin wrapper over a sherpasidecar.Client that owns the
// underlying sherpa C++ object in a sidecar subprocess.
package emotion

import (
	"fmt"

	"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// sidecarClient is the minimal interface the Classifier needs. Kept
// concrete-type-friendly with a separate ClassifyResult shape so test
// fakes don't have to import sherpasidecar.
type sidecarClient interface {
	LoadClassifier(modelDir string, threads int) error
	Classify(samples []float32, sampleRate int) (sherpasidecar.ClassifyResult, error)
}

// Classifier performs emotion + audio-event classification via a
// sherpasidecar client. The underlying SenseVoice model lives in the
// sidecar process.
type Classifier struct {
	client sidecarClient
}

// NewClassifier loads the SenseVoice emotion model into the sidecar and
// returns a Classifier that routes Classify calls through it. The caller
// owns the client's lifecycle.
func NewClassifier(client sidecarClient, modelDir string, threads int) (*Classifier, error) {
	if client == nil {
		return nil, fmt.Errorf("emotion: client is nil")
	}
	if err := client.LoadClassifier(modelDir, threads); err != nil {
		return nil, fmt.Errorf("load emotion model: %w", err)
	}
	return &Classifier{client: client}, nil
}

// Classify performs emotion classification on audio samples.
// Returns the EmotionResult (3-layer mapping), audio event string, and error.
func (c *Classifier) Classify(samples []float32, sampleRate int) (types.EmotionResult, string, error) {
	raw, err := c.client.Classify(samples, sampleRate)
	if err != nil {
		return types.EmotionResult{}, "", err
	}

	info := types.LookupEmotion(raw.Raw, types.SenseVoiceEmotionMap)

	return types.EmotionResult{
		Raw:        info.Raw,
		Label:      info.Label,
		Display:    info.Display,
		Confidence: raw.Confidence,
	}, raw.AudioEvent, nil
}

// Close is a no-op kept for API compatibility. The sidecar client owns
// the sherpa model's lifetime, not the Classifier.
func (c *Classifier) Close() {}
