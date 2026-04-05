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
	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/emotion"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/output"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newTranscribeCmd() *cobra.Command {
	var (
		inputPath   string
		outputPath  string
		format      string
		language    string
		threshold   float32
		numSpeakers int
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
			fmt.Fprintf(os.Stderr, "[1/8] Extracting embedded binaries...\n")
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// 4. Ensure ASR model
			fmt.Fprintf(os.Stderr, "[2/8] Ensuring ASR model...\n")
			asrModelName := models.ResolveASRModel(language)
			asrModelPath, err := models.EnsureModel(asrModelName)
			if err != nil {
				return fmt.Errorf("ensure ASR model: %w", err)
			}

			// 5. Ensure VAD model
			fmt.Fprintf(os.Stderr, "[3/8] Ensuring VAD model...\n")
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

			fmt.Fprintf(os.Stderr, "[4/8] Converting audio to WAV...\n")
			tempWavPath := filepath.Join(tmpDir, "audio.wav")
			if err := audio.ConvertToWAV(bins.FFmpeg, inputPath, tempWavPath); err != nil {
				return fmt.Errorf("convert to WAV: %w", err)
			}

			// 7. Run ASR
			fmt.Fprintf(os.Stderr, "[5/8] Running speech recognition...\n")
			whisperCfg := asr.WhisperConfig{
				BinPath:      bins.WhisperCLI,
				ModelPath:    asrModelPath,
				VADModelPath: vadModelPath,
				Language:     language,
				Threads:      cfg.Threads,
			}
			results, err := asr.TranscribeWithCache(whisperCfg, tempWavPath)
			if err != nil {
				return fmt.Errorf("transcribe: %w", err)
			}

			// 8. Ensure diarization models
			fmt.Fprintf(os.Stderr, "[6/8] Running speaker diarization...\n")
			segModelDir, err := models.EnsureModel("pyannote-segmentation-3-0")
			if err != nil {
				return fmt.Errorf("ensure segmentation model: %w", err)
			}
			diarEmbModelPath, err := models.EnsureModel("eres2net-embedding")
			if err != nil {
				return fmt.Errorf("ensure diarization embedding model: %w", err)
			}

			// 9. Read full WAV
			wavSamples, wavSampleRate, err := audio.ReadWAV(tempWavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}

			// 10. Run diarization
			diarizer, err := diarize.NewDiarizer(segModelDir, diarEmbModelPath, cfg.Threads, numSpeakers, threshold)
			if err != nil {
				return fmt.Errorf("init diarizer: %w", err)
			}
			defer diarizer.Close()

			diarSegments := diarizer.Process(wavSamples)

			// 11. Assign speakers to ASR segments
			speakerIDs := diarize.AssignSpeakers(results, diarSegments)

			// 12. Resolve speaker names
			fmt.Fprintf(os.Stderr, "[7/8] Resolving speaker identities...\n")
			speakerModelPath, err := models.EnsureModel("campplus-sv-zh-cn")
			if err != nil {
				return fmt.Errorf("ensure speaker model: %w", err)
			}
			extractor, err := speaker.NewExtractor(speakerModelPath, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init speaker extractor: %w", err)
			}
			defer extractor.Close()

			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})

			speakerNames, err := diarize.ResolveSpeakerNames(
				speakerIDs, diarSegments, wavSamples, wavSampleRate,
				extractor, profiles, matcher, float32(cfg.Threshold), store,
			)
			if err != nil {
				return fmt.Errorf("resolve speaker names: %w", err)
			}

			// 13. Emotion classification + build segments
			emotionModelDir, err := models.EnsureModel("sensevoice-small-int8")
			if err != nil {
				return fmt.Errorf("ensure emotion model: %w", err)
			}
			classifier, err := emotion.NewClassifier(emotionModelDir, cfg.Threads)
			if err != nil {
				return fmt.Errorf("init emotion classifier: %w", err)
			}
			defer classifier.Close()

			segments := make([]types.TranscriptSegment, 0, len(results))
			for i, r := range results {
				segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

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
					Speaker:    speakerNames[i],
					Emotion:    emotionInfo,
					AudioEvent: audioEvent,
					Language:   r.Language,
					Text:       r.Text,
					Confidence: types.Confidence{Speaker: 0, Emotion: emotionConf},
				})
			}

			// 14. Build metadata with speaker counts
			speakerSet := make(map[string]bool)
			identified := 0
			for _, seg := range segments {
				speakerSet[seg.Speaker] = true
				if !strings.HasPrefix(seg.Speaker, "speaker_") && seg.Speaker != "Unknown" {
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

			// 15. Format and write output files
			fmt.Fprintf(os.Stderr, "[8/8] Writing output files...\n")
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
	cmd.Flags().Float32Var(&threshold, "threshold", 0.5, "diarization clustering threshold (higher = fewer speakers)")
	cmd.Flags().IntVar(&numSpeakers, "num-speakers", 0, "expected number of speakers (0 = auto-detect)")
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
