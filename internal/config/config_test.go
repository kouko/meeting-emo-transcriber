package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()
	if cfg.Language != "auto" {
		t.Errorf("Language = %q, want auto", cfg.Language)
	}
	if cfg.Threshold != 0.6 {
		t.Errorf("Threshold = %f, want 0.6", cfg.Threshold)
	}
	if cfg.Format != "txt" {
		t.Errorf("Format = %q, want txt", cfg.Format)
	}
	if cfg.Strategy != "max_similarity" {
		t.Errorf("Strategy = %q, want max_similarity", cfg.Strategy)
	}
	if !cfg.Discover {
		t.Error("Discover should default to true")
	}
}

func TestLoad_FromSpeakersDir(t *testing.T) {
	dir := t.TempDir()
	yaml := "language: \"zh-TW\"\nthreshold: 0.8\nformat: json\n"
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(yaml), 0644)

	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Language != "zh-TW" {
		t.Errorf("Language = %q, want zh-TW", cfg.Language)
	}
	if cfg.Threshold != 0.8 {
		t.Errorf("Threshold = %f, want 0.8", cfg.Threshold)
	}
	if cfg.Format != "json" {
		t.Errorf("Format = %q, want json", cfg.Format)
	}
	if !cfg.Discover {
		t.Error("Discover should default to true when not in config")
	}
}

func TestLoad_NoConfigFile(t *testing.T) {
	dir := t.TempDir()
	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Language != "auto" {
		t.Errorf("Language = %q, want auto", cfg.Language)
	}
}

func TestLoad_ExplicitConfigPath(t *testing.T) {
	dir := t.TempDir()
	yaml := "language: \"ja\"\nthreshold: 0.7\n"
	configPath := filepath.Join(dir, "my-config.yaml")
	os.WriteFile(configPath, []byte(yaml), 0644)

	cfg, err := Load(configPath, "")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Language != "ja" {
		t.Errorf("Language = %q, want ja", cfg.Language)
	}
}

func TestParseFormats(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"txt", 1},
		{"txt,json", 2},
		{"txt,json,srt", 3},
		{"all", 3},
	}
	for _, tt := range tests {
		got := ParseFormats(tt.input)
		if len(got) != tt.want {
			t.Errorf("ParseFormats(%q) = %v (len %d), want len %d", tt.input, got, len(got), tt.want)
		}
	}
}
