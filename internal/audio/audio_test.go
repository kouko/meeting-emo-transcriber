package audio

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// --- convert.go tests ---

func TestParseFFmpegInfo(t *testing.T) {
	tests := []struct {
		name     string
		stderr   string
		wantInfo audioInfo
	}{
		{
			name: "16kHz mono WAV",
			stderr: `ffmpeg version 6.0
Input #0, wav, from 'test.wav':
  Duration: 00:00:10.00, bitrate: 256 kb/s
    Stream #0:0: Audio: pcm_s16le (araw / 0x77617261), 16000 Hz, mono, s16, 256 kb/s`,
			wantInfo: audioInfo{
				Codec:      "pcm_s16le",
				SampleRate: 16000,
				Channels:   1,
			},
		},
		{
			name: "44100 stereo MP3",
			stderr: `ffmpeg version 6.0
Input #0, mp3, from 'test.mp3':
  Duration: 00:03:30.00, bitrate: 128 kb/s
    Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s`,
			wantInfo: audioInfo{
				Codec:      "mp3",
				SampleRate: 44100,
				Channels:   2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseFFmpegInfo(tt.stderr)
			if got.Codec != tt.wantInfo.Codec {
				t.Errorf("Codec: got %q, want %q", got.Codec, tt.wantInfo.Codec)
			}
			if got.SampleRate != tt.wantInfo.SampleRate {
				t.Errorf("SampleRate: got %d, want %d", got.SampleRate, tt.wantInfo.SampleRate)
			}
			if got.Channels != tt.wantInfo.Channels {
				t.Errorf("Channels: got %d, want %d", got.Channels, tt.wantInfo.Channels)
			}
		})
	}
}

func TestIsTargetFormat(t *testing.T) {
	tests := []struct {
		name string
		info audioInfo
		want bool
	}{
		{
			name: "correct format",
			info: audioInfo{Codec: "pcm_s16le", SampleRate: 16000, Channels: 1},
			want: true,
		},
		{
			name: "wrong sample rate",
			info: audioInfo{Codec: "pcm_s16le", SampleRate: 44100, Channels: 1},
			want: false,
		},
		{
			name: "wrong channels",
			info: audioInfo{Codec: "pcm_s16le", SampleRate: 16000, Channels: 2},
			want: false,
		},
		{
			name: "wrong codec",
			info: audioInfo{Codec: "mp3", SampleRate: 16000, Channels: 1},
			want: false,
		},
		{
			name: "empty",
			info: audioInfo{},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isTargetFormat(tt.info); got != tt.want {
				t.Errorf("isTargetFormat(%+v) = %v, want %v", tt.info, got, tt.want)
			}
		})
	}
}

func TestBuildFFmpegArgs(t *testing.T) {
	args := buildFFmpegArgs("/usr/bin/ffmpeg", "input.mp3", "output.wav")

	// Verify all required flags are present
	flagMap := make(map[string]bool)
	for _, arg := range args {
		flagMap[arg] = true
	}

	required := []string{"-y", "-i", "-acodec", "-ar", "-ac"}
	for _, flag := range required {
		if !flagMap[flag] {
			t.Errorf("missing flag %q in args: %v", flag, args)
		}
	}

	// No -af when no filters
	if flagMap["-af"] {
		t.Errorf("should not have -af when no filters: %v", args)
	}

	// Verify input and output paths appear
	found := false
	for _, arg := range args {
		if arg == "input.mp3" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("input path not found in args: %v", args)
	}

	found = false
	for _, arg := range args {
		if arg == "output.wav" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("output path not found in args: %v", args)
	}

	// Verify specific values for pcm_s16le, 16000, 1
	for i, arg := range args {
		if arg == "-acodec" && i+1 < len(args) && args[i+1] != "pcm_s16le" {
			t.Errorf("-acodec value: got %q, want %q", args[i+1], "pcm_s16le")
		}
		if arg == "-ar" && i+1 < len(args) && args[i+1] != "16000" {
			t.Errorf("-ar value: got %q, want %q", args[i+1], "16000")
		}
		if arg == "-ac" && i+1 < len(args) && args[i+1] != "1" {
			t.Errorf("-ac value: got %q, want %q", args[i+1], "1")
		}
	}
}

func TestBuildFFmpegArgsWithFilters(t *testing.T) {
	// Single filter
	args := buildFFmpegArgs("/usr/bin/ffmpeg", "in.wav", "out.wav", "volume=-5.0dB")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-af volume=-5.0dB") {
		t.Errorf("expected -af with volume filter, got: %s", joined)
	}

	// Multiple filters should be comma-joined
	args = buildFFmpegArgs("/usr/bin/ffmpeg", "in.wav", "out.wav", "volume=-5.0dB", "loudnorm=I=-16")
	joined = strings.Join(args, " ")
	if !strings.Contains(joined, "-af volume=-5.0dB,loudnorm=I=-16") {
		t.Errorf("expected comma-joined filters, got: %s", joined)
	}
}

func TestDetectMaxVolumeRegex(t *testing.T) {
	tests := []struct {
		name    string
		output  string
		want    float64
		wantErr bool
	}{
		{
			name:   "positive peak (clipping)",
			output: "[Parsed_volumedetect_0 @ 0x...] max_volume: 8.4 dB",
			want:   8.4,
		},
		{
			name:   "negative peak (normal)",
			output: "[Parsed_volumedetect_0 @ 0x...] max_volume: -5.2 dB",
			want:   -5.2,
		},
		{
			name:   "zero peak",
			output: "[Parsed_volumedetect_0 @ 0x...] max_volume: 0.0 dB",
			want:   0.0,
		},
		{
			name:    "no match",
			output:  "some random ffmpeg output",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := maxVolumeRe.FindStringSubmatch(tt.output)
			if tt.wantErr {
				if m != nil {
					t.Error("expected no match, got match")
				}
				return
			}
			if m == nil {
				t.Fatal("expected match, got nil")
			}
			got, err := strconv.ParseFloat(m[1], 64)
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %.1f, want %.1f", got, tt.want)
			}
		})
	}
}

// --- wav.go tests ---

func TestReadWriteWAV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wav")

	// Generate 16000 samples: one second of 440 Hz sine wave at 16kHz
	sampleRate := 16000
	numSamples := sampleRate
	samples := make([]float32, numSamples)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(sampleRate)))
	}

	if err := WriteWAV(path, samples, sampleRate); err != nil {
		t.Fatalf("WriteWAV failed: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("output file not created: %v", err)
	}

	readSamples, readRate, err := ReadWAV(path)
	if err != nil {
		t.Fatalf("ReadWAV failed: %v", err)
	}

	if readRate != sampleRate {
		t.Errorf("sample rate: got %d, want %d", readRate, sampleRate)
	}

	if len(readSamples) != numSamples {
		t.Errorf("sample count: got %d, want %d", len(readSamples), numSamples)
	}

	// Verify approximate values (int16 quantization ~0.0001 tolerance not enough; use 1/32767 ≈ 0.00004)
	tolerance := float32(2.0 / 32767.0) // 2 LSB tolerance
	for i, orig := range samples {
		diff := readSamples[i] - orig
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Errorf("sample[%d]: got %f, want %f (diff %f > tolerance %f)", i, readSamples[i], orig, diff, tolerance)
			break
		}
	}
}

func TestExtractSegment(t *testing.T) {
	sampleRate := 16000
	// 2 seconds of samples: 32000 samples
	// Fill with index/32000 so values are distinguishable
	samples := make([]float32, 32000)
	for i := range samples {
		samples[i] = float32(i) / 32000.0
	}

	// Extract 0.5s to 1.5s: samples[8000:24000]
	seg := ExtractSegment(samples, sampleRate, 0.5, 1.5)

	expectedLen := 16000
	if len(seg) != expectedLen {
		t.Errorf("length: got %d, want %d", len(seg), expectedLen)
	}

	// First value should be samples[8000]
	expectedFirst := float32(8000) / 32000.0
	if seg[0] != expectedFirst {
		t.Errorf("first value: got %f, want %f", seg[0], expectedFirst)
	}
}

func TestExtractSegmentBounds(t *testing.T) {
	sampleRate := 16000
	samples := make([]float32, 16000) // 1 second

	// Beyond end: end clamps to len(samples)
	seg := ExtractSegment(samples, sampleRate, 0.0, 5.0)
	if len(seg) != 16000 {
		t.Errorf("clamped end: got %d samples, want %d", len(seg), 16000)
	}

	// Start beyond end: should return empty
	seg = ExtractSegment(samples, sampleRate, 5.0, 10.0)
	if len(seg) != 0 {
		t.Errorf("start beyond end: got %d samples, want 0", len(seg))
	}
}
