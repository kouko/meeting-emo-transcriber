package speaker

import (
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
)

// Extractor wraps sherpa-onnx SpeakerEmbeddingExtractor for CAM++ model.
type Extractor struct {
	inner *sherpa.SpeakerEmbeddingExtractor
	dim   int
}

// NewExtractor creates a speaker embedding extractor using the CAM++ ONNX model.
func NewExtractor(modelPath string, threads int) (*Extractor, error) {
	config := &sherpa.SpeakerEmbeddingExtractorConfig{}
	config.Model = modelPath
	config.NumThreads = threads
	config.Debug = 0
	config.Provider = "cpu"

	inner := sherpa.NewSpeakerEmbeddingExtractor(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create speaker embedding extractor from %s", modelPath)
	}

	dim := inner.Dim()
	return &Extractor{inner: inner, dim: dim}, nil
}

// Extract computes a speaker embedding from audio samples.
// samples must be float32 PCM at the given sampleRate.
// Returns a 512-dimensional float32 embedding vector.
func (e *Extractor) Extract(samples []float32, sampleRate int) ([]float32, error) {
	stream := e.inner.CreateStream()
	defer sherpa.DeleteOnlineStream(stream)

	stream.AcceptWaveform(sampleRate, samples)

	if !e.inner.IsReady(stream) {
		return nil, fmt.Errorf("not enough audio data for embedding extraction")
	}

	embedding := e.inner.Compute(stream)
	return embedding, nil
}

// Dim returns the embedding dimension (512 for CAM++).
func (e *Extractor) Dim() int {
	return e.dim
}

// Close releases the underlying sherpa-onnx resources.
func (e *Extractor) Close() {
	if e.inner != nil {
		sherpa.DeleteSpeakerEmbeddingExtractor(e.inner)
		e.inner = nil
	}
}
