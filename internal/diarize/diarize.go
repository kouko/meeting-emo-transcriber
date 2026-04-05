package diarize

import (
	"fmt"
	"path/filepath"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
)

// Segment represents a speaker diarization result.
type Segment struct {
	Start   float64
	End     float64
	Speaker int // 0-indexed cluster ID
}

// Diarizer wraps sherpa-onnx OfflineSpeakerDiarization.
type Diarizer struct {
	inner *sherpa.OfflineSpeakerDiarization
}

// NewDiarizer creates a diarizer.
// segModelDir: path to directory containing pyannote model.onnx
// embModelPath: path to speaker embedding .onnx file
// threads: number of CPU threads to use
// numClusters: if > 0, use fixed cluster count; otherwise use threshold
// threshold: clustering threshold (higher = fewer clusters); ignored if numClusters > 0
func NewDiarizer(segModelDir, embModelPath string, threads int, numClusters int, threshold float32) (*Diarizer, error) {
	config := &sherpa.OfflineSpeakerDiarizationConfig{}

	config.Segmentation.Pyannote.Model = filepath.Join(segModelDir, "model.onnx")
	config.Segmentation.NumThreads = threads
	config.Segmentation.Provider = "cpu"

	config.Embedding.Model = embModelPath
	config.Embedding.NumThreads = threads
	config.Embedding.Provider = "cpu"

	if numClusters > 0 {
		config.Clustering.NumClusters = numClusters
	} else {
		config.Clustering.Threshold = threshold
	}

	inner := sherpa.NewOfflineSpeakerDiarization(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create diarizer (check model paths: seg=%s, emb=%s)", segModelDir, embModelPath)
	}

	return &Diarizer{inner: inner}, nil
}

// Process runs diarization on audio samples (must be at SampleRate() Hz).
// Returns segments sorted by start time.
func (d *Diarizer) Process(samples []float32) []Segment {
	raw := d.inner.Process(samples)
	segments := make([]Segment, len(raw))
	for i, r := range raw {
		segments[i] = Segment{
			Start:   float64(r.Start),
			End:     float64(r.End),
			Speaker: r.Speaker,
		}
	}
	return segments
}

// SampleRate returns the expected audio sample rate (typically 16000).
func (d *Diarizer) SampleRate() int {
	return d.inner.SampleRate()
}

// Close releases resources.
func (d *Diarizer) Close() {
	if d.inner != nil {
		sherpa.DeleteOfflineSpeakerDiarization(d.inner)
		d.inner = nil
	}
}
