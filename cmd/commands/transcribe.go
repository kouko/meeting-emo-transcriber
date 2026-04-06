package commands

import (
	"fmt"
	"os"
	"os/exec"
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
		inputPath      string
		outputPath     string
		format         string
		language       string
		threshold      float32
		matchThreshold float32
		numSpeakers    int
		learn          bool
		enhance        bool
		prompt         string
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
			fmt.Fprintf(os.Stderr, "[1/9] Extracting embedded binaries...\n")
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// 4. Ensure ASR model
			fmt.Fprintf(os.Stderr, "[2/9] Ensuring ASR model...\n")
			asrModelName := models.ResolveASRModel(language)
			asrModelPath, err := models.EnsureModel(asrModelName)
			if err != nil {
				return fmt.Errorf("ensure ASR model: %w", err)
			}

			// 5. Ensure VAD model
			fmt.Fprintf(os.Stderr, "[3/9] Ensuring VAD model...\n")
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

			fmt.Fprintf(os.Stderr, "[4/9] Converting audio to WAV...\n")
			tempWavPath := filepath.Join(tmpDir, "audio.wav")
			if err := audio.ConvertToWAV(bins.FFmpeg, inputPath, tempWavPath); err != nil {
				return fmt.Errorf("convert to WAV: %w", err)
			}

			// Optional: enhance audio with DeepFilterNet3
			if enhance {
				fmt.Fprintf(os.Stderr, "[*] Enhancing audio (DeepFilterNet3)...\n")
				enhancedPath := filepath.Join(tmpDir, "enhanced.wav")
				cmd := exec.Command(bins.Denoise, tempWavPath, enhancedPath)
				cmd.Stderr = os.Stderr
				if err := cmd.Run(); err != nil {
					return fmt.Errorf("enhance audio: %w", err)
				}
				// Use enhanced audio for subsequent steps
				tempWavPath = enhancedPath
			}

			// 7. Run ASR
			fmt.Fprintf(os.Stderr, "[5/9] Running speech recognition...\n")
			// Merge --prompt + config vocabulary
			allPrompt := mergePrompts(prompt, cfg.Vocabulary)
			if allPrompt != "" {
				fmt.Fprintf(os.Stderr, "  --prompt=%q\n", allPrompt)
			}

			whisperCfg := asr.WhisperConfig{
				BinPath:      bins.WhisperCLI,
				ModelPath:    asrModelPath,
				VADModelPath: vadModelPath,
				Language:     language,
				Threads:      cfg.Threads,
				Prompt:       allPrompt,
			}
			results, err := asr.TranscribeWithCache(whisperCfg, tempWavPath, inputPath)
			if err != nil {
				return fmt.Errorf("transcribe: %w", err)
			}

			// 8. Run diarization (FluidAudio subprocess, includes speaker embeddings)
			speakersDesc := "auto"
			if numSpeakers > 0 {
				speakersDesc = fmt.Sprintf("%d", numSpeakers)
			}
			fmt.Fprintf(os.Stderr, "[6/9] Running speaker diarization...\n")
			fmt.Fprintf(os.Stderr, "  --threshold=%.2f (higher=more speakers, lower=fewer speakers)\n", threshold)
			fmt.Fprintf(os.Stderr, "  --num-speakers=%s\n", speakersDesc)
			diarResult, err := diarize.Process(bins.Diarize, tempWavPath, threshold, numSpeakers)
			if err != nil {
				return fmt.Errorf("diarization: %w", err)
			}

			// 9. Read full WAV for segment extraction
			wavSamples, wavSampleRate, err := audio.ReadWAV(tempWavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}

			// 10. Assign speakers to ASR segments
			speakerIDs := diarize.AssignSpeakers(results, diarResult.Segments)

			// 11. Resolve speaker names (WeSpeaker 256-dim centroid embeddings)
			fmt.Fprintf(os.Stderr, "[7/9] Resolving speaker identities...\n")
			fmt.Fprintf(os.Stderr, "  --match-threshold=%.2f (higher=stricter matching, lower=more lenient)\n", matchThreshold)
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())

			// Auto-enroll: extract voiceprint from concatenated wav
			extractFn := func(wavPath string) ([]float32, error) {
				result, err := diarize.ExtractVoiceprint(bins.Diarize, wavPath)
				if err != nil {
					return nil, err
				}
				emb := make([]float32, len(result.Vector))
				for i, v := range result.Vector {
					emb[i] = float32(v)
				}
				return emb, nil
			}
			if enrolled, err := speaker.AutoEnroll(store, bins.FFmpeg, extractFn); err != nil {
				return fmt.Errorf("auto-enroll: %w", err)
			} else if enrolled > 0 {
				fmt.Fprintf(os.Stderr, "  Auto-enrolled %d speaker(s)\n", enrolled)
			}

			profiles, err := store.LoadProfiles()
			if err != nil {
				return fmt.Errorf("load speaker profiles: %w", err)
			}

			speakerNames, err := diarize.ResolveSpeakerNames(
				speakerIDs, diarResult, wavSamples, wavSampleRate,
				profiles, matchThreshold, store, bins.Diarize, learn,
			)
			if err != nil {
				return fmt.Errorf("resolve speaker names: %w", err)
			}

			// 13. Emotion classification + build segments
			fmt.Fprintf(os.Stderr, "[8/9] Running emotion classification...\n")
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
			fmt.Fprintf(os.Stderr, "[9/9] Writing output files...\n")
			formats := config.ParseFormats(format)
			for _, fmt_ := range formats {
				outPath := resolveOutputPath(inputPath, outputPath, fmt_)
				content, err := formatTranscript(fmt_, transcript)
				if err != nil {
					return fmt.Errorf("format %s: %w", fmt_, err)
				}
				// Add UTF-8 BOM for txt/srt so macOS text editors detect encoding correctly
				var fileContent []byte
				if fmt_ == "txt" || fmt_ == "srt" {
					fileContent = append([]byte{0xEF, 0xBB, 0xBF}, []byte(content)...)
				} else {
					fileContent = []byte(content)
				}
				if err := os.WriteFile(outPath, fileContent, 0644); err != nil {
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
	cmd.Flags().Float32Var(&threshold, "threshold", 0.8, "diarization clustering threshold (higher = more speakers)")
	cmd.Flags().Float32Var(&matchThreshold, "match-threshold", 0.55, "speaker matching threshold for enrolled profiles")
	cmd.Flags().IntVar(&numSpeakers, "num-speakers", 0, "expected number of speakers (0 = auto-detect)")
	cmd.Flags().BoolVarP(&learn, "learning-mode", "l", false, "create folders for all clusters (including matched) for manual review")
	cmd.Flags().BoolVar(&enhance, "enhance", false, "enhance audio with DeepFilterNet3 noise reduction before processing")
	cmd.Flags().StringVar(&prompt, "prompt", "", "custom vocabulary/context hints for ASR (comma-separated)")
	cmd.MarkFlagRequired("input")
	return cmd
}

// resolveOutputPath determines the output file path for a given format.
// mergePrompts combines CLI --prompt and config.yaml vocabulary into one string.
func mergePrompts(cliPrompt string, configVocab []string) string {
	var parts []string
	if len(configVocab) > 0 {
		parts = append(parts, strings.Join(configVocab, ", "))
	}
	if cliPrompt != "" {
		parts = append(parts, cliPrompt)
	}
	return strings.Join(parts, ", ")
}

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
