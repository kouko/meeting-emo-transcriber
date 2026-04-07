package emotion

import (
	"errors"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar"
)

// fakeClient is a test double that records load calls and returns canned
// responses. It lets us exercise the Classifier wrapper without spawning
// a real sidecar.
type fakeClient struct {
	loadErr     error
	classifyErr error
	classifyRes sherpasidecar.ClassifyResult

	loadedModelDir string
	loadedThreads  int
}

func (f *fakeClient) LoadClassifier(modelDir string, threads int) error {
	f.loadedModelDir = modelDir
	f.loadedThreads = threads
	return f.loadErr
}

func (f *fakeClient) Classify(samples []float32, sampleRate int) (sherpasidecar.ClassifyResult, error) {
	if f.classifyErr != nil {
		return sherpasidecar.ClassifyResult{}, f.classifyErr
	}
	return f.classifyRes, nil
}

func TestNewClassifierPropagatesLoadError(t *testing.T) {
	fake := &fakeClient{loadErr: errors.New("boom")}
	if _, err := NewClassifier(fake, "/tmp/whatever", 1); err == nil {
		t.Error("expected error when LoadClassifier returns error")
	}
	if fake.loadedModelDir != "/tmp/whatever" || fake.loadedThreads != 1 {
		t.Errorf("load args not forwarded: got modelDir=%q threads=%d", fake.loadedModelDir, fake.loadedThreads)
	}
}

func TestClassifierMapsEmotionRaw(t *testing.T) {
	fake := &fakeClient{
		classifyRes: sherpasidecar.ClassifyResult{
			Raw:        "HAPPY",
			AudioEvent: "Speech",
			Confidence: 0.9,
		},
	}
	cls, err := NewClassifier(fake, "/tmp/x", 2)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}

	result, event, err := cls.Classify([]float32{0.1, 0.2}, 16000)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if result.Label != "Happy" {
		t.Errorf("Label = %q, want Happy", result.Label)
	}
	if result.Display != "happily" {
		t.Errorf("Display = %q, want happily", result.Display)
	}
	if result.Confidence != 0.9 {
		t.Errorf("Confidence = %v, want 0.9", result.Confidence)
	}
	if event != "Speech" {
		t.Errorf("event = %q, want Speech", event)
	}
}

func TestClassifierPropagatesClassifyError(t *testing.T) {
	fake := &fakeClient{classifyErr: errors.New("stream dead")}
	cls, err := NewClassifier(fake, "/tmp/x", 1)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	if _, _, err := cls.Classify(nil, 16000); err == nil {
		t.Error("expected error to propagate")
	}
}
