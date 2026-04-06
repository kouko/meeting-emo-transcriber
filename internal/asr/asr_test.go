package asr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestParseSRTTimestamp tests 7 valid timestamp cases.
func TestParseSRTTimestamp(t *testing.T) {
	cases := []struct {
		input    string
		expected float64
	}{
		{"00:00:00,000", 0.0},
		{"00:00:01,000", 1.0},
		{"00:00:01,500", 1.5},
		{"00:01:00,000", 60.0},
		{"01:00:00,000", 3600.0},
		{"01:23:45,678", 5025.678},
		{"00:00:00,001", 0.001},
	}

	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			got, err := ParseSRTTimestamp(tc.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Use a small epsilon for float comparison
			const eps = 1e-9
			diff := got - tc.expected
			if diff < -eps || diff > eps {
				t.Errorf("ParseSRTTimestamp(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}

// TestParseSRTTimestampInvalid tests invalid timestamp formats.
func TestParseSRTTimestampInvalid(t *testing.T) {
	invalids := []string{
		"",
		"00:00:00",
		"abc",
		"00:00:00.000",
		"1:2:3,4",
	}

	for _, ts := range invalids {
		t.Run(ts, func(t *testing.T) {
			_, err := ParseSRTTimestamp(ts)
			if err == nil {
				t.Errorf("ParseSRTTimestamp(%q) expected error, got nil", ts)
			}
		})
	}
}

// TestParseSRT tests a 3-block SRT with mixed English/Chinese content.
func TestParseSRT(t *testing.T) {
	content := `1
00:00:01,000 --> 00:00:03,500
Hello, world.

2
00:00:04,000 --> 00:00:06,000
This is a test.

3
00:00:07,000 --> 00:00:09,000
你好，世界。
`

	results, err := ParseSRT(content, "en")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// Block 1
	r := results[0]
	if r.Start != 1.0 {
		t.Errorf("results[0].Start = %v, want 1.0", r.Start)
	}
	if r.End != 3.5 {
		t.Errorf("results[0].End = %v, want 3.5", r.End)
	}
	if r.Text != "Hello, world." {
		t.Errorf("results[0].Text = %q, want %q", r.Text, "Hello, world.")
	}
	if r.Language != "en" {
		t.Errorf("results[0].Language = %q, want %q", r.Language, "en")
	}

	// Block 2
	r = results[1]
	if r.Start != 4.0 {
		t.Errorf("results[1].Start = %v, want 4.0", r.Start)
	}
	if r.End != 6.0 {
		t.Errorf("results[1].End = %v, want 6.0", r.End)
	}
	if r.Text != "This is a test." {
		t.Errorf("results[1].Text = %q, want %q", r.Text, "This is a test.")
	}

	// Block 3 (Chinese)
	r = results[2]
	if r.Start != 7.0 {
		t.Errorf("results[2].Start = %v, want 7.0", r.Start)
	}
	if r.End != 9.0 {
		t.Errorf("results[2].End = %v, want 9.0", r.End)
	}
	if r.Text != "你好，世界。" {
		t.Errorf("results[2].Text = %q, want %q", r.Text, "你好，世界。")
	}
	if r.Language != "en" {
		t.Errorf("results[2].Language = %q, want %q", r.Language, "en")
	}
}

// TestParseSRTMultilineText tests that multi-line subtitles are joined with spaces.
func TestParseSRTMultilineText(t *testing.T) {
	content := `1
00:00:01,000 --> 00:00:05,000
Line one of the subtitle.
Line two of the subtitle.
`

	results, err := ParseSRT(content, "zh")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	want := "Line one of the subtitle. Line two of the subtitle."
	if results[0].Text != want {
		t.Errorf("Text = %q, want %q", results[0].Text, want)
	}

	if results[0].Language != "zh" {
		t.Errorf("Language = %q, want %q", results[0].Language, "zh")
	}
}

// TestParseSRTEmpty tests that empty string returns nil, no error.
func TestParseSRTEmpty(t *testing.T) {
	results, err := ParseSRT("", "en")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for empty input, got %v", results)
	}
}

// TestParseSRTWhitespace tests that whitespace-only string returns nil, no error.
func TestParseSRTWhitespace(t *testing.T) {
	results, err := ParseSRT("   \n\t\n   ", "en")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for whitespace input, got %v", results)
	}
}

func TestBuildWhisperArgs(t *testing.T) {
	cfg := WhisperConfig{
		BinPath:   "/path/to/whisper-cli",
		ModelPath: "/path/to/model.bin",
		Language:  "auto",
		Threads:   4,
	}

	args := buildWhisperArgs(cfg, "/tmp/input.wav", "/tmp/output")
	joined := strings.Join(args, " ")

	checks := []string{
		"-m /path/to/model.bin",
		"-f /tmp/input.wav",
		"-l auto",
		"-t 4",
		"-osrt",
		"-of /tmp/output",
	}
	for _, check := range checks {
		if !strings.Contains(joined, check) {
			t.Errorf("missing %q in args: %s", check, joined)
		}
	}
}

func TestBuildWhisperArgsWithPrompt(t *testing.T) {
	cfg := WhisperConfig{
		BinPath:   "/path/to/whisper-cli",
		ModelPath: "/path/to/model.bin",
		Language:  "ja",
		Threads:   8,
		Prompt:    "kouko, YanJen",
	}
	args := buildWhisperArgs(cfg, "/tmp/input.wav", "/tmp/output")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--prompt") {
		t.Error("should contain --prompt when Prompt is set")
	}
	if !strings.Contains(joined, "kouko, YanJen") {
		t.Error("should contain prompt text")
	}
}

func TestContentFingerprint(t *testing.T) {
	// Create two files with identical content at different paths
	tmpDir := t.TempDir()
	content := []byte("hello world test audio content for fingerprint")

	pathA := filepath.Join(tmpDir, "a.wav")
	pathB := filepath.Join(tmpDir, "subdir", "b.wav")
	os.MkdirAll(filepath.Dir(pathB), 0755)
	os.WriteFile(pathA, content, 0644)
	os.WriteFile(pathB, content, 0644)

	fpA, err := contentFingerprint(pathA)
	if err != nil {
		t.Fatalf("fingerprint A: %v", err)
	}
	fpB, err := contentFingerprint(pathB)
	if err != nil {
		t.Fatalf("fingerprint B: %v", err)
	}

	if fpA != fpB {
		t.Errorf("same content at different paths should produce same fingerprint: %q != %q", fpA, fpB)
	}

	// Different content should produce different fingerprint
	pathC := filepath.Join(tmpDir, "c.wav")
	os.WriteFile(pathC, []byte("different content entirely"), 0644)
	fpC, err := contentFingerprint(pathC)
	if err != nil {
		t.Fatalf("fingerprint C: %v", err)
	}
	if fpA == fpC {
		t.Error("different content should produce different fingerprint")
	}
}

func TestCacheKeyFactors(t *testing.T) {
	tmpDir := t.TempDir()
	wavPath := filepath.Join(tmpDir, "test.wav")
	os.WriteFile(wavPath, []byte("test audio data for cache key"), 0644)

	baseCfg := WhisperConfig{
		BinPath:   "/bin/whisper",
		ModelPath: "/models/ggml-large-v3.bin",
		Language:  "auto",
		Threads:   4,
		Prompt:    "",
	}

	baseKey, err := cacheKey(wavPath, baseCfg)
	if err != nil {
		t.Fatalf("base cache key: %v", err)
	}

	// Different language -> different key
	langCfg := baseCfg
	langCfg.Language = "ja"
	langKey, _ := cacheKey(wavPath, langCfg)
	if baseKey == langKey {
		t.Error("different language should produce different cache key")
	}

	// Different prompt -> same key (prompt is intentionally excluded)
	promptCfg := baseCfg
	promptCfg.Prompt = "kouko, YanJen"
	promptKey, _ := cacheKey(wavPath, promptCfg)
	if baseKey != promptKey {
		t.Error("different prompt should produce same cache key (prompt excluded)")
	}

	// Same config -> same key (deterministic)
	sameKey, _ := cacheKey(wavPath, baseCfg)
	if baseKey != sameKey {
		t.Error("same inputs should produce same cache key")
	}
}
