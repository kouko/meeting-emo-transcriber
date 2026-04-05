package speaker

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewExtractorRequiresModel(t *testing.T) {
	_, err := NewExtractor("/nonexistent/model.onnx", 1)
	if err == nil {
		t.Error("expected error for non-existent model path")
	}
}

func TestExtractorDim(t *testing.T) {
	modelPath := findTestModel("campplus-sv-zh-cn")
	if modelPath == "" {
		t.Skip("CAM++ model not available, skipping")
	}

	ext, err := NewExtractor(modelPath, 1)
	if err != nil {
		t.Fatalf("NewExtractor: %v", err)
	}
	defer ext.Close()

	dim := ext.Dim()
	if dim <= 0 {
		t.Errorf("Dim() = %d, want > 0", dim)
	}
	t.Logf("Dim() = %d", dim)
}

func findTestModel(name string) string {
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".metr", "models", name+".onnx"),
		filepath.Join(home, ".metr", "models", name+".bin"),
		filepath.Join(home, ".metr", "models", name),
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}
