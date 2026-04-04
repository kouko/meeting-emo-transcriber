package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/asr"
	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/emotion"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/output"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newTranscribeCmd() *cobra.Command {
	var (
		inputPath  string
		outputPath string
		format     string
		language   string
		threshold  float32
		noDiscover bool
	)
	cmd := &cobra.Command{
		Use:   "transcribe",
		Short: "Transcribe a meeting recording",
		RunE: func(cmd *cobra.Command, args []string) error {
			// 1. Validate input file exists
			if _, err := os.Stat(inputPath); err != nil {
				return fmt.Errorf("input file not found: %w", err)
			}

			// 2. Load config
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}

			// 3. Extract embedded binaries
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// 4. Ensure ASR model
			asrModelName := models.ResolveASRModel(language)
			asrModelPath, err := models.EnsureModel(asrModelName)
			if err != nil {
				return fmt.Errorf("ensure ASR model: %w", err)
			}

			// 5. Ensure VAD model
			vadModelPath, err := models.EnsureModel("silero-vad-v6.2.0")
			if err != nil {
				return fmt.Errorf("ensure VAD model: %w", err)
			}

			// 6. Create temp dir and convert to WAV
			tmpDir, err := os.MkdirTemp("", "met-transcribe-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			tempWavPath := filepath.Join(tmpDir, "audio.wav")
			if err := audio.ConvertToWAV(bins.FFmpeg, inputPath, tempWavPath); err != nil {
				return fmt.Errorf("convert to WAV: %w", err)
			}

			// 7. Run ASR
			whisperCfg := asr.WhisperConfig{
				BinPath:      bins.WhisperCLI,
				ModelPath:    asrModelPath,
				VADModelPath: vadModelPath,
				Language:     language,
				Threads:      cfg.Threads,
			}
			results, err := asr.Transcribe(whisperCfg, tempWavPath)
			if err != nil {
				return fmt.Errorf("transcribe: %w", err)
			}

			// 8. Ensure speaker and emotion models
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}
			emotionModelDir, err := models.EnsureModel("sensevoice-small-int8")
			if err != nil {
				return fmt.Errorf("ensure emotion model: %w", err)
			}

			// 9. Initialize extractor and classifier
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init speaker extractor: %w", err)
			}
			defer extractor.Close()

			classifier, err := emotion.NewClassifier(emotionModelDir, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init emotion classifier: %w", err)
			}
			defer classifier.Close()

			// 10. Load speaker profiles + create matcher
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})
			discovery := speaker.NewDiscovery(store, extractor, matcher, float32(cfg.Threshold), noDiscover)

			// 11. Read full WAV for segment extraction
			wavSamples, wavSampleRate, err := audio.ReadWAV(tempWavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}

			// 12. Process each ASR segment
			segments := make([]types.TranscriptSegment, 0, len(results))
			for _, r := range results {
				segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

				// Speaker identification
				speakerName := "Unknown"
				var speakerConf float32
				if len(segAudio) > 0 {
					emb, embErr := extractor.Extract(segAudio, wavSampleRate)
					if embErr == nil {
						speakerName, speakerConf = discovery.IdentifySpeaker(
							emb, profiles, segAudio, wavSampleRate, r.Start,
						)
					}
				}

				// Emotion classification
				emotionInfo := types.EmotionInfo{Label: "Neutral", Display: ""}
				audioEvent := "Speech"
				var emotionConf float32
				if len(segAudio) > 0 {
					emotionResult, event, classErr := classifier.Classify(segAudio, wavSampleRate)
					if classErr == nil {
						emotionInfo = types.EmotionInfo{
							Raw:     emotionResult.Raw,
							Label:   emotionResult.Label,
							Display: emotionResult.Display,
						}
						audioEvent = event
						emotionConf = emotionResult.Confidence
					}
				}

				segments = append(segments, types.TranscriptSegment{
					Start:      r.Start,
					End:        r.End,
					Speaker:    speakerName,
					Emotion:    emotionInfo,
					AudioEvent: audioEvent,
					Language:   r.Language,
					Text:       r.Text,
					Confidence: types.Confidence{Speaker: speakerConf, Emotion: emotionConf},
				})
			}

			// 13. Build metadata with speaker counts
			speakerSet := make(map[string]bool)
			identified := 0
			for _, seg := range segments {
				speakerSet[seg.Speaker] = true
				if seg.Speaker != "Unknown" &&
					!strings.HasPrefix(seg.Speaker, "speaker_") &&
					!strings.HasPrefix(seg.Speaker, "Unknown_") {
					identified++
				}
			}

			// Calculate duration from last result's End time
			var duration float64
			if len(results) > 0 {
				duration = results[len(results)-1].End
			}

			transcript := types.TranscriptResult{
				Metadata: types.Metadata{
					File:               filepath.Base(inputPath),
					Duration:           time.Duration(duration * float64(time.Second)).String(),
					SpeakersDetected:   len(speakerSet),
					SpeakersIdentified: identified,
					Date:               time.Now().Format(time.RFC3339),
				},
				Segments: segments,
			}

			// 9 & 10. Format and write output files
			formats := config.ParseFormats(format)
			for _, fmt_ := range formats {
				outPath := resolveOutputPath(inputPath, outputPath, fmt_)
				content, err := formatTranscript(fmt_, transcript)
				if err != nil {
					return fmt.Errorf("format %s: %w", fmt_, err)
				}
				if err := os.WriteFile(outPath, []byte(content), 0644); err != nil {
					return fmt.Errorf("write %s: %w", outPath, err)
				}
				fmt.Fprintf(os.Stderr, "Written: %s\n", outPath)
			}

			return nil
		},
	}
	cmd.Flags().StringVar(&inputPath, "input", "", "input audio file path (required)")
	cmd.Flags().StringVar(&outputPath, "output", "", "output file path")
	cmd.Flags().StringVar(&format, "format", "txt", "output format: txt|json|srt|all (comma-separated)")
	cmd.Flags().StringVar(&language, "language", "auto", "language: auto|zh-TW|zh|en|ja")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.6, "speaker similarity threshold")
	cmd.Flags().BoolVar(&noDiscover, "no-discover", false, "disable unknown speaker auto-discovery")
	cmd.MarkFlagRequired("input")
	return cmd
}

// resolveOutputPath determines the output file path for a given format.
func resolveOutputPath(inputPath, outputPath, format string) string {
	ext := "." + format
	if outputPath != "" {
		base := strings.TrimSuffix(outputPath, filepath.Ext(outputPath))
		return base + ext
	}
	base := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))
	return base + ext
}

// formatTranscript dispatches to the appropriate formatter.
func formatTranscript(format string, result types.TranscriptResult) (string, error) {
	switch format {
	case "json":
		f := &output.JSONFormatter{}
		return f.Format(result)
	case "srt":
		f := &output.SRTFormatter{}
		return f.Format(result)
	default: // "txt" and anything else
		f := &output.TXTFormatter{}
		return f.Format(result)
	}
}
