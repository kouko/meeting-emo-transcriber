package audio

import (
	"fmt"
	"math"
	"os"

	goaudio "github.com/go-audio/audio"
	"github.com/go-audio/wav"
)

// ReadWAV reads a WAV file and returns float32 samples normalized to [-1,1] and the sample rate.
func ReadWAV(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("open WAV: %w", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		return nil, 0, fmt.Errorf("decode WAV: %w", err)
	}

	bitDepth := int(dec.BitDepth)
	if bitDepth == 0 {
		bitDepth = 16
	}
	scale := float32(math.Pow(2, float64(bitDepth-1)))

	samples := make([]float32, len(buf.Data))
	for i, v := range buf.Data {
		samples[i] = float32(v) / scale
	}

	return samples, int(dec.SampleRate), nil
}

// WriteWAV writes float32 samples (normalized [-1,1]) to a 16-bit PCM WAV file.
func WriteWAV(path string, samples []float32, sampleRate int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create WAV: %w", err)
	}
	defer f.Close()

	enc := wav.NewEncoder(f, sampleRate, 16, 1, 1)

	const scale = float32(1 << 15) // 32768
	data := make([]int, len(samples))
	for i, s := range samples {
		// Clamp to [-1, 1]
		if s > 1.0 {
			s = 1.0
		} else if s < -1.0 {
			s = -1.0
		}
		v := int(s * scale)
		// Clamp to int16 range
		if v > math.MaxInt16 {
			v = math.MaxInt16
		} else if v < math.MinInt16 {
			v = math.MinInt16
		}
		data[i] = v
	}

	buf := &goaudio.IntBuffer{
		Data: data,
		Format: &goaudio.Format{
			SampleRate:  sampleRate,
			NumChannels: 1,
		},
		SourceBitDepth: 16,
	}

	if err := enc.Write(buf); err != nil {
		return fmt.Errorf("write WAV: %w", err)
	}
	return enc.Close()
}

// RMS computes the root-mean-square of float32 audio samples.
// Returns 0.0 for empty input.
func RMS(samples []float32) float64 {
	if len(samples) == 0 {
		return 0.0
	}
	var sum float64
	for _, s := range samples {
		sum += float64(s) * float64(s)
	}
	return math.Sqrt(sum / float64(len(samples)))
}

// ExtractSegment returns the samples between start and end seconds.
// Indices are clamped to valid bounds. If start >= len(samples), returns empty slice.
func ExtractSegment(samples []float32, sampleRate int, start, end float64) []float32 {
	total := len(samples)

	startIdx := int(start * float64(sampleRate))
	endIdx := int(end * float64(sampleRate))

	// Clamp
	if startIdx < 0 {
		startIdx = 0
	}
	if startIdx >= total {
		return []float32{}
	}
	if endIdx > total {
		endIdx = total
	}
	if endIdx <= startIdx {
		return []float32{}
	}

	seg := make([]float32, endIdx-startIdx)
	copy(seg, samples[startIdx:endIdx])
	return seg
}
