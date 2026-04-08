package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/viper"
)

// Models holds paths for the three ML model files.
type Models struct {
	Whisper  string
	Speaker  string
	Emotion  string
}

// Config holds all application configuration.
type Config struct {
	Language       string
	Threshold      float64
	MatchThreshold float64
	Format         string
	Strategy       string
	Discover       bool
	LogLevel       string
	Threads        int
	Models         Models
	Vocabulary        []string // custom vocabulary for whisper prompt
	MinSampleDuration float64  // minimum segment duration (seconds) for auto-discovered speaker samples
	MinSampleRMS      float64  // minimum RMS energy for speaker samples (0.0-1.0)
}

// defaultModelsDir returns ~/.metr/models/
func defaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".metr", "models")
}

// Defaults returns a Config populated with sensible defaults.
func Defaults() Config {
	modelsDir := defaultModelsDir()
	return Config{
		Language:       "auto",
		Threshold:      0.8,
		MatchThreshold: 0.55,
		Format:         "txt",
		Strategy:  "max_similarity",
		Discover:          true,
		MinSampleDuration: 15.0,
		MinSampleRMS:      0.01,
		LogLevel:  "info",
		Threads:   runtime.NumCPU(),
		Models: Models{
			Whisper:  "",
			Speaker:  filepath.Join(modelsDir, "campplus_sv_zh-cn.onnx"),
			Emotion:  filepath.Join(modelsDir, "sensevoice-small-int8.onnx"),
		},
	}
}

// Load reads configuration with the following priority (highest to lowest):
//  1. Explicit config file at configPath
//  2. speakers/config.yaml inside speakersDir
//  3. Built-in defaults
//
// Either configPath or speakersDir may be empty string.
func Load(configPath, speakersDir string) (Config, error) {
	v := viper.New()

	// Set defaults.
	d := Defaults()
	v.SetDefault("language", d.Language)
	v.SetDefault("threshold", d.Threshold)
	v.SetDefault("match_threshold", d.MatchThreshold)
	v.SetDefault("format", d.Format)
	v.SetDefault("strategy", d.Strategy)
	v.SetDefault("discover", d.Discover)
	v.SetDefault("loglevel", d.LogLevel)
	v.SetDefault("threads", d.Threads)
	v.SetDefault("min_sample_duration", d.MinSampleDuration)
	v.SetDefault("min_sample_rms", d.MinSampleRMS)
	v.SetDefault("models.whisper", d.Models.Whisper)
	v.SetDefault("models.speaker", d.Models.Speaker)
	v.SetDefault("models.emotion", d.Models.Emotion)

	if configPath != "" {
		// Explicit config path takes highest priority.
		v.SetConfigFile(configPath)
		if err := v.ReadInConfig(); err != nil {
			return Config{}, err
		}
	} else if speakersDir != "" {
		// Try speakers/_metr/config.yaml first, fallback to speakers/config.yaml
		v.SetConfigName("config")
		v.SetConfigType("yaml")
		v.AddConfigPath(filepath.Join(speakersDir, "_metr"))
		v.AddConfigPath(speakersDir) // backward compat
		// Ignore error when file doesn't exist.
		_ = v.ReadInConfig()
	}

	cfg := Config{
		Language:       v.GetString("language"),
		Threshold:      v.GetFloat64("threshold"),
		MatchThreshold: v.GetFloat64("match_threshold"),
		Format:         v.GetString("format"),
		Strategy:   v.GetString("strategy"),
		Discover:   v.GetBool("discover"),
		LogLevel:   v.GetString("loglevel"),
		Threads:    v.GetInt("threads"),
		Vocabulary:        v.GetStringSlice("vocabulary"),
		MinSampleDuration: v.GetFloat64("min_sample_duration"),
		MinSampleRMS:      v.GetFloat64("min_sample_rms"),
		Models: Models{
			Whisper: v.GetString("models.whisper"),
			Speaker: v.GetString("models.speaker"),
			Emotion: v.GetString("models.emotion"),
		},
	}
	return cfg, nil
}

// SaveableConfig holds user-facing settings to write to config.yaml.
type SaveableConfig struct {
	Language       string   `yaml:"language"`
	Threshold      float64  `yaml:"threshold"`
	MatchThreshold float64  `yaml:"match_threshold"`
	Format         string   `yaml:"format"`
	Vocabulary        []string `yaml:"vocabulary,omitempty"`
	MinSampleDuration float64  `yaml:"min_sample_duration"`
	MinSampleRMS      float64  `yaml:"min_sample_rms"`
}

// Save writes user-facing config values to a YAML file.
// If the file does not exist, a commented template is generated.
// If the file already exists, only the values are updated (preserving comments).
func Save(configPath string, sc SaveableConfig) error {
	_, err := os.Stat(configPath)
	if err != nil {
		// First time: generate full template with comments
		return writeConfigTemplate(configPath, sc)
	}
	// Already exists: update values only
	return writeConfigCompact(configPath, sc)
}

// writeConfigTemplate generates a config.yaml with detailed comments.
func writeConfigTemplate(configPath string, sc SaveableConfig) error {
	var lines []string
	lines = append(lines, "# metr config.yaml")
	lines = append(lines, "# -------------------------------------------------------")
	lines = append(lines, "# Settings here serve as defaults; CLI flags override them.")
	lines = append(lines, "# Only explicitly-set CLI flags will update this file.")
	lines = append(lines, "# -------------------------------------------------------")
	lines = append(lines, "")
	lines = append(lines, `# ASR language (default: "auto")`)
	lines = append(lines, "# Determines which whisper model to use:")
	lines = append(lines, `#   "auto"  -> ggml-large-v3 (multilingual, auto-detect)`)
	lines = append(lines, `#   "en"    -> ggml-large-v3`)
	lines = append(lines, `#   "zh-TW" -> ggml-breeze-asr-25-q5k (Traditional Chinese, optimized)`)
	lines = append(lines, `#   "zh"    -> ggml-belle-zh (Simplified Chinese)`)
	lines = append(lines, `#   "ja"    -> ggml-kotoba-whisper-v2.0 (Japanese)`)
	lines = append(lines, fmt.Sprintf("language: %q", sc.Language))
	lines = append(lines, "")
	lines = append(lines, "# Diarization clustering threshold (default: 0.80)")
	lines = append(lines, "# Controls how aggressively speakers are split during diarization.")
	lines = append(lines, "#   Higher value (e.g. 0.9) -> more speakers detected (finer splitting)")
	lines = append(lines, "#   Lower value  (e.g. 0.5) -> fewer speakers detected (more merging)")
	lines = append(lines, "# Range: 0.0 - 1.0")
	lines = append(lines, fmt.Sprintf("threshold: %.2f", sc.Threshold))
	lines = append(lines, "")
	lines = append(lines, "# Speaker matching threshold, cosine similarity (default: 0.55)")
	lines = append(lines, "# Used when matching diarized clusters to enrolled speaker profiles.")
	lines = append(lines, "#   Higher value (e.g. 0.7) -> stricter matching, fewer false positives")
	lines = append(lines, "#   Lower value  (e.g. 0.4) -> more lenient, may match wrong speakers")
	lines = append(lines, "# Range: 0.0 - 1.0")
	lines = append(lines, fmt.Sprintf("match_threshold: %.2f", sc.MatchThreshold))
	lines = append(lines, "")
	lines = append(lines, `# Output format (default: "txt")`)
	lines = append(lines, `# Supported: "txt", "json", "srt", "all" (generates all three)`)
	lines = append(lines, `# Comma-separated for multiple: "txt,json"`)
	lines = append(lines, fmt.Sprintf("format: %q", sc.Format))
	lines = append(lines, "")
	lines = append(lines, "# Custom vocabulary / context hints for ASR (default: empty)")
	lines = append(lines, "# Improves recognition of proper nouns, technical terms, etc.")
	lines = append(lines, "# Each entry is passed to whisper's --prompt as comma-separated hints.")
	lines = append(lines, "# CLI --prompt values are merged with this list (no duplicates).")
	lines = append(lines, "")
	lines = append(lines, "# Minimum segment duration for auto-discovered speaker samples (default: 15.0)")
	lines = append(lines, "# Speakers whose longest segment is shorter than this are marked Unknown.")
	lines = append(lines, "# Higher values require longer continuous speech for reliable voiceprint enrollment.")
	lines = append(lines, fmt.Sprintf("min_sample_duration: %.1f", sc.MinSampleDuration))
	lines = append(lines, "")
	lines = append(lines, "# Minimum RMS energy for speaker samples (default: 0.01)")
	lines = append(lines, "# Segments below this energy level are considered silence/noise and skipped.")
	lines = append(lines, "# Range: 0.0 - 1.0 (on normalized [-1,1] audio)")
	lines = append(lines, fmt.Sprintf("min_sample_rms: %.2f", sc.MinSampleRMS))
	if len(sc.Vocabulary) > 0 {
		lines = append(lines, "vocabulary:")
		for _, v := range sc.Vocabulary {
			lines = append(lines, fmt.Sprintf("  - %q", v))
		}
	} else {
		lines = append(lines, "vocabulary: []")
		lines = append(lines, `#   - "Alice"`)
		lines = append(lines, `#   - "ACME Corp"`)
	}
	lines = append(lines, "")

	content := strings.Join(lines, "\n")
	return os.WriteFile(configPath, []byte(content), 0644)
}

// writeConfigCompact updates an existing config.yaml with minimal formatting.
func writeConfigCompact(configPath string, sc SaveableConfig) error {
	var lines []string
	lines = append(lines, fmt.Sprintf("language: %q", sc.Language))
	lines = append(lines, fmt.Sprintf("threshold: %.2f", sc.Threshold))
	lines = append(lines, fmt.Sprintf("match_threshold: %.2f", sc.MatchThreshold))
	lines = append(lines, fmt.Sprintf("format: %q", sc.Format))
	lines = append(lines, fmt.Sprintf("min_sample_duration: %.1f", sc.MinSampleDuration))
	lines = append(lines, fmt.Sprintf("min_sample_rms: %.2f", sc.MinSampleRMS))
	if len(sc.Vocabulary) > 0 {
		lines = append(lines, "vocabulary:")
		for _, v := range sc.Vocabulary {
			lines = append(lines, fmt.Sprintf("  - %q", v))
		}
	}
	lines = append(lines, "")

	content := strings.Join(lines, "\n")
	return os.WriteFile(configPath, []byte(content), 0644)
}

// allFormats is the canonical list of supported output formats.
var allFormats = []string{"txt", "json", "srt"}

// ParseFormats parses a comma-separated format string into a slice of format names.
// The special value "all" expands to all supported formats.
func ParseFormats(format string) []string {
	if strings.EqualFold(strings.TrimSpace(format), "all") {
		result := make([]string, len(allFormats))
		copy(result, allFormats)
		return result
	}
	parts := strings.Split(format, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// SupportedAudioExtensions returns the list of audio file extensions this tool handles.
func SupportedAudioExtensions() []string {
	return []string{".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".mkv", ".webm"}
}
