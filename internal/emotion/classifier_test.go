package emotion

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewClassifierRequiresModel(t *testing.T) {
	_, err := NewClassifier("/nonexistent/model", 1)
	if err == nil {
		t.Error("expected error for non-existent model path")
	}
}

func TestClassifyWithModel(t *testing.T) {
	modelDir := findSenseVoiceModel()
	if modelDir == "" {
		t.Skip("SenseVoice model not available, skipping")
	}

	cls, err := NewClassifier(modelDir, 1)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer cls.Close()

	// 1 second of silence at 16kHz
	samples := make([]float32, 16000)
	result, event, err := cls.Classify(samples, 16000)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	if result.Label == "" {
		t.Error("expected non-empty emotion label")
	}
	t.Logf("Emotion: %+v, Event: %s", result, event)
}

func findSenseVoiceModel() string {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".meeting-emo-transcriber", "models", "sensevoice-small-int8")
	if _, err := os.Stat(filepath.Join(dir, "model.int8.onnx")); err == nil {
		return dir
	}
	return ""
}
