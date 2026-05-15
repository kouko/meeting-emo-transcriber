package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaults(t *testing.T) {
	cfg := Defaults()
	if cfg.Language != "auto" {
		t.Errorf("Language = %q, want auto", cfg.Language)
	}
	if cfg.Threshold != 0.8 {
		t.Errorf("Threshold = %f, want 0.8", cfg.Threshold)
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
	if cfg.MinSampleDuration != 15.0 {
		t.Errorf("MinSampleDuration = %f, want 15.0", cfg.MinSampleDuration)
	}
	if cfg.MinSampleRMS != 0.01 {
		t.Errorf("MinSampleRMS = %f, want 0.01", cfg.MinSampleRMS)
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

func TestLoad_MinSampleFields(t *testing.T) {
	dir := t.TempDir()
	yaml := "min_sample_duration: 20.0\nmin_sample_rms: 0.05\n"
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(yaml), 0644)

	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MinSampleDuration != 20.0 {
		t.Errorf("MinSampleDuration = %f, want 20.0", cfg.MinSampleDuration)
	}
	if cfg.MinSampleRMS != 0.05 {
		t.Errorf("MinSampleRMS = %f, want 0.05", cfg.MinSampleRMS)
	}
}

func TestLoad_MinSampleFieldsDefault(t *testing.T) {
	dir := t.TempDir()
	// config.yaml without min_sample fields — should use defaults
	yaml := "language: \"en\"\n"
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(yaml), 0644)

	cfg, err := Load("", dir)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MinSampleDuration != 15.0 {
		t.Errorf("MinSampleDuration = %f, want 15.0 (default)", cfg.MinSampleDuration)
	}
	if cfg.MinSampleRMS != 0.01 {
		t.Errorf("MinSampleRMS = %f, want 0.01 (default)", cfg.MinSampleRMS)
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

func TestSave_FirstTimeWritesTemplate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	sc := SaveableConfig{
		Language:          "zh-TW",
		Threshold:         0.8,
		MatchThreshold:    0.65,
		MatchMargin:       0.07,
		Format:            "txt",
		MinSampleDuration: 15.0,
		MinSampleRMS:      0.01,
	}
	if err := Save(path, sc); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)

	// Template should contain helpful comments and the actual values.
	for _, must := range []string{
		"# metr config.yaml",
		"language: \"zh-TW\"",
		"match_threshold: 0.65",
		"match_margin: 0.07",
		"min_sample_duration: 15.0",
		"# Speaker matching threshold",
		"# Speaker matching margin",
	} {
		if !strings.Contains(content, must) {
			t.Errorf("template missing %q\n--- got ---\n%s", must, content)
		}
	}
}

func TestSave_ExistingFileWritesCompact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	// Pre-create the file so Save takes the compact path.
	if err := os.WriteFile(path, []byte("# existing\n"), 0644); err != nil {
		t.Fatal(err)
	}

	sc := SaveableConfig{
		Language:          "en",
		Threshold:         0.9,
		MatchThreshold:    0.70,
		MatchMargin:       0.10,
		Format:            "json",
		MinSampleDuration: 20.0,
		MinSampleRMS:      0.02,
		Vocabulary:        []string{"Alice", "ACME"},
	}
	if err := Save(path, sc); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, _ := os.ReadFile(path)
	content := string(data)

	// Compact form drops the long header comments but keeps each key.
	if strings.Contains(content, "# metr config.yaml") {
		t.Errorf("compact path should not write the template banner; got:\n%s", content)
	}
	for _, must := range []string{
		`language: "en"`,
		"threshold: 0.90",
		"match_threshold: 0.70",
		"match_margin: 0.10",
		`format: "json"`,
		"min_sample_duration: 20.0",
		"min_sample_rms: 0.02",
		`- "Alice"`,
		`- "ACME"`,
	} {
		if !strings.Contains(content, must) {
			t.Errorf("compact form missing %q\n--- got ---\n%s", must, content)
		}
	}
}

func TestSave_RoundTrip(t *testing.T) {
	// Write a SaveableConfig via Save, then load it via Load and confirm
	// every persisted field is recovered.
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")

	sc := SaveableConfig{
		Language:          "ja",
		Threshold:         0.75,
		MatchThreshold:    0.62,
		MatchMargin:       0.05,
		Format:            "srt",
		MinSampleDuration: 12.0,
		MinSampleRMS:      0.03,
		Vocabulary:        []string{"Hayashi"},
	}
	if err := Save(path, sc); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Language != sc.Language {
		t.Errorf("Language = %q, want %q", cfg.Language, sc.Language)
	}
	if cfg.Threshold != sc.Threshold {
		t.Errorf("Threshold = %f, want %f", cfg.Threshold, sc.Threshold)
	}
	if cfg.MatchThreshold != sc.MatchThreshold {
		t.Errorf("MatchThreshold = %f, want %f", cfg.MatchThreshold, sc.MatchThreshold)
	}
	if cfg.MatchMargin != sc.MatchMargin {
		t.Errorf("MatchMargin = %f, want %f", cfg.MatchMargin, sc.MatchMargin)
	}
	if cfg.Format != sc.Format {
		t.Errorf("Format = %q, want %q", cfg.Format, sc.Format)
	}
	if cfg.MinSampleDuration != sc.MinSampleDuration {
		t.Errorf("MinSampleDuration = %f, want %f", cfg.MinSampleDuration, sc.MinSampleDuration)
	}
	if len(cfg.Vocabulary) != len(sc.Vocabulary) || cfg.Vocabulary[0] != "Hayashi" {
		t.Errorf("Vocabulary = %v, want %v", cfg.Vocabulary, sc.Vocabulary)
	}
}

func TestSupportedAudioExtensions(t *testing.T) {
	got := SupportedAudioExtensions()
	wantSet := map[string]bool{".wav": true, ".mp3": true, ".m4a": true, ".flac": true}
	for ext := range wantSet {
		found := false
		for _, g := range got {
			if g == ext {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("SupportedAudioExtensions missing %q (got %v)", ext, got)
		}
	}
	// All extensions should start with a dot.
	for _, ext := range got {
		if len(ext) < 2 || ext[0] != '.' {
			t.Errorf("extension %q must start with a dot", ext)
		}
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
