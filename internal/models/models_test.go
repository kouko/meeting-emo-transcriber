package models

import (
	"strings"
	"testing"
)

func TestResolveASRModel(t *testing.T) {
	cases := []string{"auto", "en", "zh-TW", "zh", "ja", "unknown"}
	for _, lang := range cases {
		t.Run(lang, func(t *testing.T) {
			got := ResolveASRModel(lang)
			if got != "ggml-large-v3" {
				t.Errorf("ResolveASRModel(%q) = %q; want %q", lang, got, "ggml-large-v3")
			}
		})
	}
}

func TestRegistryContainsRequiredModels(t *testing.T) {
	required := []string{"ggml-large-v3", "silero-vad-v6.2.0"}
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
