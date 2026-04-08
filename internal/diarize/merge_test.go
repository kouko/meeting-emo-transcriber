package diarize

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// generateSineWAV creates float32 samples of a sine wave at the given frequency.
func generateSineWAV(sampleRate int, durationSec float64, amplitude float32) []float32 {
	n := int(float64(sampleRate) * durationSec)
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = amplitude * float32(math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)))
	}
	return samples
}

func TestPersistUnknownSpeaker_RMSFilter(t *testing.T) {
	dir := t.TempDir()
	store := speaker.NewStore(dir, config.SupportedAudioExtensions())
	sampleRate := 16000

	// 60 seconds of audio: first 20s loud, next 20s silent, last 20s loud
	loud := generateSineWAV(sampleRate, 20.0, 0.5)
	silent := make([]float32, sampleRate*20)
	wavSamples := append(append(loud, silent...), loud...)

	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},   // loud, RMS >> 0.01
			{Start: 20, End: 40, Speaker: "C1"},   // silent, RMS = 0
			{Start: 40, End: 60, Speaker: "C1"},   // loud, RMS >> 0.01
		},
	}

	persistUnknownSpeaker(store, "test_speaker", "C1", diarResult, wavSamples, sampleRate, 0.01)

	// Should save 2 WAV files (the two loud segments), not the silent one
	speakerDir := filepath.Join(dir, "test_speaker")
	entries, err := os.ReadDir(speakerDir)
	if err != nil {
		t.Fatalf("read speaker dir: %v", err)
	}

	wavCount := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".wav" {
			wavCount++
		}
	}
	if wavCount != 2 {
		t.Errorf("WAV files saved = %d, want 2 (silent segment should be filtered)", wavCount)
	}
}

func TestPersistUnknownSpeaker_AllSilent(t *testing.T) {
	dir := t.TempDir()
	store := speaker.NewStore(dir, config.SupportedAudioExtensions())
	sampleRate := 16000

	// 30 seconds of silence
	wavSamples := make([]float32, sampleRate*30)

	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 15, Speaker: "C1"},
			{Start: 15, End: 30, Speaker: "C1"},
		},
	}

	persistUnknownSpeaker(store, "silent_speaker", "C1", diarResult, wavSamples, sampleRate, 0.01)

	// Should save 0 WAV files, but profile.json should still exist
	speakerDir := filepath.Join(dir, "silent_speaker")
	entries, err := os.ReadDir(speakerDir)
	if err != nil {
		t.Fatalf("read speaker dir: %v", err)
	}

	wavCount := 0
	profileCount := 0
	for _, e := range entries {
		switch filepath.Ext(e.Name()) {
		case ".wav":
			wavCount++
		case ".json":
			profileCount++
		}
	}
	if wavCount != 0 {
		t.Errorf("WAV files saved = %d, want 0 (all silent)", wavCount)
	}
	if profileCount != 1 {
		t.Errorf("profile files = %d, want 1", profileCount)
	}
}

func TestResolveSpeakerNames_ShortSegmentsMarkedUnknown(t *testing.T) {
	dir := t.TempDir()
	store := speaker.NewStore(dir, config.SupportedAudioExtensions())
	sampleRate := 16000

	// 30 seconds of audio (enough samples)
	wavSamples := generateSineWAV(sampleRate, 30.0, 0.5)

	// Cluster C1 has longest segment 20s (above threshold)
	// Cluster C2 has longest segment 5s (below threshold of 15s)
	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 20, Speaker: "C1"},
			{Start: 20, End: 25, Speaker: "C2"},
			{Start: 25, End: 30, Speaker: "C2"},
		},
		SpeakerVoiceprints: map[string][]float64{},
	}

	speakerIDs := []string{"C1", "C2", "C2"}

	names, err := ResolveSpeakerNames(
		speakerIDs, diarResult, wavSamples, sampleRate,
		[]types.SpeakerProfile{}, // no enrolled profiles
		0.55, store, "", false,
		15.0, // minSampleDuration
		0.01, // minSampleRMS
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}

	// C1 (20s) → speaker_1
	if names[0] != "speaker_1" {
		t.Errorf("names[0] = %q, want speaker_1 (20s segment should qualify)", names[0])
	}
	// C2 (5s max) → Unknown
	if names[1] != "Unknown" {
		t.Errorf("names[1] = %q, want Unknown (5s < 15s threshold)", names[1])
	}
	if names[2] != "Unknown" {
		t.Errorf("names[2] = %q, want Unknown (5s < 15s threshold)", names[2])
	}

	// speaker_1 dir should exist, no speaker dir for Unknown
	if _, err := os.Stat(filepath.Join(dir, "speaker_1")); os.IsNotExist(err) {
		t.Error("speaker_1 directory should exist")
	}
}

func TestResolveSpeakerNames_ZeroThresholdAllowsAll(t *testing.T) {
	dir := t.TempDir()
	store := speaker.NewStore(dir, config.SupportedAudioExtensions())
	sampleRate := 16000

	wavSamples := generateSineWAV(sampleRate, 10.0, 0.5)

	diarResult := &DiarizeResult{
		Segments: []Segment{
			{Start: 0, End: 3, Speaker: "C1"},
		},
		SpeakerVoiceprints: map[string][]float64{},
	}

	speakerIDs := []string{"C1"}

	names, err := ResolveSpeakerNames(
		speakerIDs, diarResult, wavSamples, sampleRate,
		[]types.SpeakerProfile{}, 0.55, store, "", false,
		0.0,  // minSampleDuration = 0 → allow all
		0.01,
	)
	if err != nil {
		t.Fatalf("ResolveSpeakerNames: %v", err)
	}

	// Even 3s segment should get speaker_1 when threshold is 0
	if names[0] != "speaker_1" {
		t.Errorf("names[0] = %q, want speaker_1 (minSampleDuration=0 allows all)", names[0])
	}
}
