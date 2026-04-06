package emotion

import (
	"fmt"
	"path/filepath"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// Classifier wraps sherpa-onnx OfflineRecognizer with SenseVoice model
// for emotion classification and audio event detection.
type Classifier struct {
	inner *sherpa.OfflineRecognizer
}

// NewClassifier creates an emotion classifier from a SenseVoice model directory.
// modelDir should contain model.int8.onnx and tokens.txt.
func NewClassifier(modelDir string, threads int) (*Classifier, error) {
	modelPath := filepath.Join(modelDir, "model.int8.onnx")
	tokensPath := filepath.Join(modelDir, "tokens.txt")

	config := &sherpa.OfflineRecognizerConfig{}
	config.ModelConfig.SenseVoice.Model = modelPath
	config.ModelConfig.SenseVoice.Language = ""
	config.ModelConfig.SenseVoice.UseInverseTextNormalization = 0
	config.ModelConfig.Tokens = tokensPath
	config.ModelConfig.NumThreads = threads
	config.ModelConfig.Debug = 0
	config.ModelConfig.Provider = "cpu"

	inner := sherpa.NewOfflineRecognizer(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create emotion classifier from %s", modelDir)
	}

	return &Classifier{inner: inner}, nil
}

// Classify performs emotion classification on audio samples.
// Returns the EmotionResult (3-layer mapping), audio event string, and error.
func (c *Classifier) Classify(samples []float32, sampleRate int) (types.EmotionResult, string, error) {
	stream := sherpa.NewOfflineStream(c.inner)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)
	c.inner.Decode(stream)
	result := stream.GetResult()

	if result == nil {
		return types.EmotionResult{}, "", fmt.Errorf("no result from emotion classifier")
	}

	// SenseVoice returns tags like "<|HAPPY|>" and "<|Speech|>" — strip markers
	emotionRaw := strings.TrimPrefix(strings.TrimSuffix(result.Emotion, "|>"), "<|")
	audioEvent := strings.TrimPrefix(strings.TrimSuffix(result.Event, "|>"), "<|")
	if audioEvent == "" {
		audioEvent = "Speech"
	}

	info := types.LookupEmotion(emotionRaw, types.SenseVoiceEmotionMap)

	return types.EmotionResult{
		Raw:        info.Raw,
		Label:      info.Label,
		Display:    info.Display,
		Confidence: 0,
	}, audioEvent, nil
}

// Close releases the underlying sherpa-onnx resources.
func (c *Classifier) Close() {
	if c.inner != nil {
		sherpa.DeleteOfflineRecognizer(c.inner)
		c.inner = nil
	}
}
