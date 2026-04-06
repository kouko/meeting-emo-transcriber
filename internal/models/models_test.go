package models

import (
	"strings"
	"testing"
)

func TestResolveASRModel(t *testing.T) {
	tests := []struct {
		lang, want string
	}{
		{"auto", "ggml-large-v3"},
		{"en", "ggml-large-v3"},
		{"zh-TW", "ggml-breeze-asr-25-q5k"},
		{"zh", "ggml-belle-zh"},
		{"ja", "ggml-kotoba-whisper-v2.0"},
		{"unknown", "ggml-large-v3"},
	}
	for _, tt := range tests {
		t.Run(tt.lang, func(t *testing.T) {
			got := ResolveASRModel(tt.lang)
			if got != tt.want {
				t.Errorf("ResolveASRModel(%q) = %q; want %q", tt.lang, got, tt.want)
			}
		})
	}
}

func TestRegistryContainsRequiredModels(t *testing.T) {
	required := []string{"ggml-large-v3", "sensevoice-small-int8"}
	for _, name := range required {
		t.Run(name, func(t *testing.T) {
			if _, ok := Registry[name]; !ok {
				t.Errorf("Registry missing required model %q", name)
			}
		})
	}
}

func TestRegistryFieldsNotEmpty(t *testing.T) {
	for name, info := range Registry {
		t.Run(name, func(t *testing.T) {
			if strings.TrimSpace(info.URL) == "" {
				t.Errorf("model %q has empty URL", name)
			}
			if strings.TrimSpace(info.Category) == "" {
				t.Errorf("model %q has empty Category", name)
			}
			if info.Size <= 0 {
				t.Errorf("model %q has Size <= 0: %d", name, info.Size)
			}
		})
	}
}

func TestModelsDir(t *testing.T) {
	dir := ModelsDir()
	if strings.TrimSpace(dir) == "" {
		t.Error("ModelsDir() returned empty string")
	}
}

func TestRegistryContainsEmotionModel(t *testing.T) {
	info, ok := Registry["sensevoice-small-int8"]
	if !ok {
		t.Fatal("Registry missing sensevoice-small-int8")
	}
	if info.Category != "emotion" {
		t.Errorf("Category = %q, want \"emotion\"", info.Category)
	}
	if !info.IsArchive {
		t.Error("expected IsArchive = true for sensevoice")
	}
}

func TestModelFilename(t *testing.T) {
	tests := []struct {
		name, url, expected string
	}{
		{"ggml-large-v3", "https://example.com/model.bin", "ggml-large-v3.bin"},
		{"sensevoice-small-int8", "https://example.com/archive.tar.bz2", "sensevoice-small-int8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := modelFilename(tt.name, tt.url)
			if got != tt.expected {
				t.Errorf("modelFilename(%q, %q) = %q, want %q", tt.name, tt.url, got, tt.expected)
			}
		})
	}
}
