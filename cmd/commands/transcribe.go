package commands

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/asr"
	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/output"
	"github.com/kouko/meeting-emo-transcriber/internal/punctuation"
	"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newTranscribeCmd() *cobra.Command {
	var (
		inputPath                 string
		outputPath                string
		format                    string
		language                  string
		threshold                 float32
		matchThreshold            float32
		matchMargin               float32
		numSpeakers               int
		dryRun                    bool
		noDiscover                bool
		learn                     bool
		enhance                   bool
		normalize                 bool
		noCache                   bool
		prompt                    string
		minSampleDuration         float64
		minSampleRMS              float64
		verifySegments            bool
		verifySegmentsThreshold   float32
		verifySegmentsMinDuration float64
	)
	cmd := &cobra.Command{
		Use:   "transcribe --input <audio file>",
		Short: "Transcribe a meeting recording",
		Long: `Transcribe a meeting recording with speaker diarization and emotion recognition.

Shortcut: metr meeting.mp3  (auto-detected by file extension)

Examples:
  metr transcribe --input meeting.mp3
  metr transcribe --input meeting.mp3 -l zh-TW --format all
  metr transcribe --input meeting.mp3 --enhance --normalize
  metr transcribe --input meeting.mp3 -L`,
		RunE: func(cmd *cobra.Command, args []string) error {
			// 1. Validate input file exists
			if _, err := os.Stat(inputPath); err != nil {
				return fmt.Errorf("input file not found: %w", err)
			}

			// 2. Load config, then overlay only explicitly-set CLI flags
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}
			if cmd.Flags().Changed("language") {
				cfg.Language = language
			}
			if cmd.Flags().Changed("threshold") {
				cfg.Threshold = float64(threshold)
			}
			if cmd.Flags().Changed("match-threshold") {
				cfg.MatchThreshold = float64(matchThreshold)
			}
			if cmd.Flags().Changed("match-margin") {
				cfg.MatchMargin = float64(matchMargin)
			}
			if cmd.Flags().Changed("format") {
				cfg.Format = format
			}
			if cmd.Flags().Changed("min-sample-duration") {
				cfg.MinSampleDuration = minSampleDuration
			}
			if cmd.Flags().Changed("min-sample-rms") {
				cfg.MinSampleRMS = minSampleRMS
			}
			if cmd.Flags().Changed("no-discover") && noDiscover {
				cfg.Discover = false
			}
			if cmd.Flags().Changed("verify-segments") {
				cfg.VerifySegments = verifySegments
			}
			if cmd.Flags().Changed("verify-threshold") {
				cfg.VerifySegmentsThreshold = float64(verifySegmentsThreshold)
			}
			if cmd.Flags().Changed("verify-min-duration") {
				cfg.VerifySegmentsMinDuration = verifySegmentsMinDuration
			}
			// Sync back to local vars so downstream code uses merged values
			language = cfg.Language
			threshold = float32(cfg.Threshold)
			matchThreshold = float32(cfg.MatchThreshold)
			matchMargin = float32(cfg.MatchMargin)
			format = cfg.Format
			minSampleDuration = cfg.MinSampleDuration
			minSampleRMS = cfg.MinSampleRMS
			verifySegments = cfg.VerifySegments
			verifySegmentsThreshold = float32(cfg.VerifySegmentsThreshold)
			verifySegmentsMinDuration = cfg.VerifySegmentsMinDuration

			// 3. Extract embedded binaries
			fmt.Fprintf(os.Stderr, "[1/9] Extracting embedded binaries...\n")
			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// 3b. Spawn sherpa-onnx sidecar. All punctuation and emotion
			// work flows through this child process, keeping the main metr
			// binary cgo-free and single-file portable. The sidecar is
			// kept alive for the entire transcribe run and closed on exit.
			sherpaClient, err := sherpasidecar.Spawn(bins.SherpaSidecar)
			if err != nil {
				return fmt.Errorf("spawn sherpa sidecar: %w", err)
			}
			defer sherpaClient.Close()

			// 4. Ensure ASR model
			var asrModelPath string
			if !dryRun {
				fmt.Fprintf(os.Stderr, "[2/9] Ensuring ASR model...\n")
				asrModelName := models.ResolveASRModel(language)
				asrModelPath, err = models.EnsureModel(asrModelName)
				if err != nil {
					return fmt.Errorf("ensure ASR model: %w", err)
				}
			} else {
				fmt.Fprintf(os.Stderr, "[dry-run] Skipping ASR model + emotion classifier — diarize + matching only\n")
			}

			// 5. Create temp dir and convert to WAV
			tmpDir, err := os.MkdirTemp("", "met-transcribe-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			fmt.Fprintf(os.Stderr, "[3/9] Converting audio to WAV...\n")
			tempWavPath := filepath.Join(tmpDir, "audio.wav")
			convOpts := audio.ConvertOpts{Normalize: normalize}
			if err := audio.ConvertToWAV(bins.FFmpeg, inputPath, tempWavPath, convOpts); err != nil {
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

			// 7. Run ASR (skipped in dry-run)
			var results []types.ASRResult
			var punctFunc func(string) string
			if !dryRun {
				fmt.Fprintf(os.Stderr, "[4/9] Running speech recognition...\n")
				// Merge --prompt + config vocabulary + enrolled speaker names
				allPrompt := mergePrompts(prompt, cfg.Vocabulary)
				if names, err := listSpeakerNames(speakersDir); err == nil && len(names) > 0 {
					allPrompt = appendSpeakerNames(allPrompt, names)
				}
				if allPrompt != "" {
					fmt.Fprintf(os.Stderr, "  --prompt=%q\n", allPrompt)
				}

				whisperCfg := asr.WhisperConfig{
					BinPath:   bins.WhisperCLI,
					ModelPath: asrModelPath,
					Language:  language,
					Threads:   cfg.Threads,
					Prompt:    allPrompt,
				}
				if noCache {
					results, err = asr.Transcribe(whisperCfg, tempWavPath)
				} else {
					results, err = asr.TranscribeWithCache(whisperCfg, tempWavPath)
				}
				if err != nil {
					return fmt.Errorf("transcribe: %w", err)
				}

				// 5. Initialize punctuator (ZH/EN only, skip JA) — used later in TXT output
				if language != "ja" {
					puncModelDir, puncErr := models.EnsureModel("ct-punc-zh-en-int8")
					if puncErr == nil {
						punc, puncInitErr := punctuation.NewPunctuator(sherpaClient, puncModelDir, cfg.Threads)
						if puncInitErr == nil {
							fmt.Fprintf(os.Stderr, "[5/9] Punctuation model loaded\n")
							defer punc.Close()
							lang := language
							punctFunc = func(text string) string {
								return punc.AddPunct(text, lang)
							}
						} else {
							puncErr = puncInitErr
						}
					}
					if puncErr != nil {
						fmt.Fprintf(os.Stderr, "  Warning: punctuation skipped: %v\n", puncErr)
					}
				}
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
			// In dry-run, we have no ASR results; treat each diarization
			// segment as a pseudo-ASR result so the matcher still sees the
			// full cluster set and prints its decisions.
			var speakerIDs []string
			if dryRun {
				results = make([]types.ASRResult, len(diarResult.Segments))
				speakerIDs = make([]string, len(diarResult.Segments))
				for i, seg := range diarResult.Segments {
					results[i] = types.ASRResult{Start: seg.Start, End: seg.End}
					speakerIDs[i] = seg.Speaker
				}
			} else {
				speakerIDs = diarize.AssignSpeakers(results, diarResult.Segments)
			}

			// 11. Resolve speaker names (WeSpeaker 256-dim centroid embeddings)
			fmt.Fprintf(os.Stderr, "[7/9] Resolving speaker identities...\n")
			fmt.Fprintf(os.Stderr, "  --match-threshold=%.2f (higher=stricter matching, lower=more lenient)\n", matchThreshold)
			fmt.Fprintf(os.Stderr, "  --match-margin=%.2f (min gap between best and runner-up profile)\n", matchMargin)
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
				profiles, matchThreshold, matchMargin, store, bins.Diarize, learn,
				minSampleDuration, minSampleRMS,
			)
			if err != nil {
				return fmt.Errorf("resolve speaker names: %w", err)
			}

			// 12b. Optional per-segment re-verification: re-check each ASR
			// segment's embedding against its matched profile and demote
			// any segment whose cosine falls below the (looser) verify
			// threshold. Catches stray foreign speech merged into a cluster.
			if verifySegments && len(profiles) > 0 {
				fmt.Fprintf(os.Stderr,
					"  --verify-segments enabled (threshold=%.2f, min-duration=%.1fs)\n",
					verifySegmentsThreshold, verifySegmentsMinDuration)
				asrResults := make([]types.ASRResult, len(results))
				copy(asrResults, results)
				batchExtract := func(wavPaths []string) ([][]float32, error) {
					vp, err := diarize.ExtractVoiceprints(bins.Diarize, wavPaths)
					if err != nil {
						return nil, err
					}
					out := make([][]float32, len(vp))
					for i, v := range vp {
						emb := make([]float32, len(v.Vector))
						for j, x := range v.Vector {
							emb[j] = float32(x)
						}
						out[i] = emb
					}
					return out, nil
				}
				verifyTmpDir, err := os.MkdirTemp("", "met-verify-*")
				if err != nil {
					return fmt.Errorf("create verify temp dir: %w", err)
				}
				refined, verifyErr := diarize.RefineSpeakerNamesPerSegment(
					speakerNames, asrResults, profiles, wavSamples, wavSampleRate,
					batchExtract, verifySegmentsThreshold, verifySegmentsMinDuration, verifyTmpDir,
				)
				os.RemoveAll(verifyTmpDir)
				if verifyErr != nil {
					fmt.Fprintf(os.Stderr, "  warning: per-segment verify failed (%v), keeping cluster labels\n", verifyErr)
				} else {
					speakerNames = refined
				}
			}

			// Dry-run: stop here. Matching decisions have been printed to
			// stderr by ResolveSpeakerNames; no transcript is produced.
			if dryRun {
				fmt.Fprintf(os.Stderr, "\nDry run complete. No transcript written.\n")
				fmt.Fprintf(os.Stderr, "Re-run without --dry-run, or with adjusted --match-threshold / --match-margin, to produce a transcript.\n")
				return nil
			}

			// 13. Emotion classification + build segments
			fmt.Fprintf(os.Stderr, "[8/9] Running emotion classification...\n")
			transcript, err := classifyEmotionsAndBuildSegments(
				results, speakerNames, wavSamples, wavSampleRate,
				sherpaClient, cfg.Threads, inputPath,
			)
			if err != nil {
				return err
			}

			// 14. Format and write output files
			fmt.Fprintf(os.Stderr, "[9/9] Writing output files...\n")
			if err := writeTranscriptOutputs(transcript, format, inputPath, outputPath, punctFunc); err != nil {
				return err
			}

			// Save current settings to config.yaml for next run
			metrDir := filepath.Join(speakersDir, "_metr")
			os.MkdirAll(metrDir, 0755)
			configSavePath := filepath.Join(metrDir, "config.yaml")
			// Merge prompt into vocabulary for saving
			var vocabToSave []string
			vocabToSave = append(vocabToSave, cfg.Vocabulary...)
			if prompt != "" {
				for _, p := range strings.Split(prompt, ",") {
					p = strings.TrimSpace(p)
					if p != "" {
						// Avoid duplicates
						found := false
						for _, v := range vocabToSave {
							if v == p {
								found = true
								break
							}
						}
						if !found {
							vocabToSave = append(vocabToSave, p)
						}
					}
				}
			}
			sc := config.SaveableConfig{
				Language:          language,
				Threshold:         float64(threshold),
				MatchThreshold:    float64(matchThreshold),
				MatchMargin:       float64(matchMargin),
				Format:            format,
				Vocabulary:        vocabToSave,
				MinSampleDuration: minSampleDuration,
				MinSampleRMS:      minSampleRMS,
			}
			if err := config.Save(configSavePath, sc); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not save config: %v\n", err)
			}

			printSpeakerGuide(speakersDir)

			return nil
		},
	}
	cmd.Flags().StringVar(&inputPath, "input", "", "input audio file path (required)")
	cmd.Flags().StringVar(&outputPath, "output", "", "output file path")
	cmd.Flags().StringVar(&format, "format", "txt", "output format: txt|json|srt|all (comma-separated)")
	cmd.Flags().StringVarP(&language, "language", "l", "auto", "language: auto|zh-TW|zh|en|ja")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.8, "diarization clustering threshold (higher = more speakers)")
	cmd.Flags().Float32Var(&matchThreshold, "match-threshold", 0.65, "speaker matching threshold for enrolled profiles")
	cmd.Flags().Float32Var(&matchMargin, "match-margin", 0.07, "minimum cosine gap between best and runner-up profile to accept a match")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "skip ASR/emotion/output — only run diarize + speaker matching (fast threshold tuning)")
	cmd.Flags().BoolVar(&noDiscover, "no-discover", false, "do not auto-create speaker_N folders for unmatched clusters (otherwise on by default; can also be set via _metr/config.yaml discover: false)")
	cmd.Flags().BoolVar(&verifySegments, "verify-segments", false, "re-verify each ASR segment against its matched profile and demote stray segments to Unknown")
	cmd.Flags().Float32Var(&verifySegmentsThreshold, "verify-threshold", 0.50, "per-segment verification threshold (looser than --match-threshold)")
	cmd.Flags().Float64Var(&verifySegmentsMinDuration, "verify-min-duration", 1.0, "minimum segment duration (seconds) to re-verify")
	cmd.Flags().IntVar(&numSpeakers, "num-speakers", 0, "expected number of speakers (0 = auto-detect)")
	cmd.Flags().BoolVarP(&learn, "learning-mode", "L", false, "create folders for all clusters (including matched) for manual review")
	cmd.Flags().BoolVar(&enhance, "enhance", false, "enhance audio with DeepFilterNet3 noise reduction before processing")
	cmd.Flags().BoolVar(&normalize, "normalize", false, "apply loudnorm normalization (auto-attenuates >0dB clipping regardless)")
	cmd.Flags().BoolVar(&noCache, "no-cache", false, "skip ASR cache and force re-transcription")
	cmd.Flags().StringVar(&prompt, "prompt", "", "custom vocabulary/context hints for ASR (comma-separated)")
	cmd.Flags().Float64Var(&minSampleDuration, "min-sample-duration", 15.0, "minimum segment duration (seconds) for auto-discovered speaker samples")
	cmd.Flags().Float64Var(&minSampleRMS, "min-sample-rms", 0.01, "minimum RMS energy for speaker samples (0.0-1.0)")
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

// listSpeakerNames returns enrolled speaker folder names, excluding
// _metr (resource dir) and speaker_ prefixed dirs (auto-generated unknowns).
func listSpeakerNames(speakersDir string) ([]string, error) {
	entries, err := os.ReadDir(speakersDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if name == "_metr" || strings.HasPrefix(name, "speaker_") {
			continue
		}
		names = append(names, name)
	}
	return names, nil
}

// appendSpeakerNames appends speaker names to an existing prompt string, avoiding duplicates.
func appendSpeakerNames(prompt string, names []string) string {
	existing := make(map[string]bool)
	for _, p := range strings.Split(prompt, ",") {
		existing[strings.TrimSpace(p)] = true
	}
	var toAdd []string
	for _, name := range names {
		if !existing[name] {
			toAdd = append(toAdd, name)
		}
	}
	if len(toAdd) == 0 {
		return prompt
	}
	addition := strings.Join(toAdd, ", ")
	if prompt == "" {
		return addition
	}
	return prompt + ", " + addition
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
// punctFunc is optional and only used by TXT formatter for post-merge punctuation.
func formatTranscript(format string, result types.TranscriptResult, punctFunc func(string) string) (string, error) {
	switch format {
	case "json":
		f := &output.JSONFormatter{}
		return f.Format(result)
	case "srt":
		f := &output.SRTFormatter{}
		return f.Format(result)
	default: // "txt" and anything else
		f := &output.TXTFormatter{PunctFunc: punctFunc}
		return f.Format(result)
	}
}

func printSpeakerGuide(speakersDir string) {
	fmt.Fprintf(os.Stderr, `
------------------------------------------------------------
Speaker Management / 話者管理 / 講者管理
------------------------------------------------------------

[EN] Open the speakers folder in Finder: %[1]s

  Rename a speaker:
    Right-click "speaker_0" folder > Rename > type the person's name
    (e.g. "Alice")

  Merge two speakers (same person detected as different speakers):
    Open "speaker_1" folder, select all files inside,
    drag & drop them into the "Alice" folder,
    then delete the empty "speaker_1" folder.

  After renaming or merging, run the same audio file again
  to get updated results with correct speaker names.

  Tip: add --learning-mode to create folders for ALL detected
  speakers (including already matched ones) for manual review.

[JA] Finder で話者フォルダを開いてください: %[1]s

  話者の名前を変更する:
    「speaker_0」フォルダを右クリック > 名前を変更 >
    本人の名前を入力（例：「Alice」）

  話者を統合する（同一人物が別々に検出された場合）:
    「speaker_1」フォルダを開き、中のファイルを全て選択して
    「Alice」フォルダにドラッグ＆ドロップし、
    空になった「speaker_1」フォルダを削除してください。

  名前変更や統合の後、同じ音声ファイルを再実行すると
  正しい話者名で結果が更新されます。

  ヒント: --learning-mode を付けると、全検出話者の
  フォルダが作成され、手動で確認できます。

[ZH] 請用 Finder 開啟講者資料夾: %[1]s

  重新命名講者:
    在「speaker_0」資料夾上按右鍵 > 重新命名 >
    輸入本人的名字（例如「Alice」）

  合併講者（同一人被辨識為不同講者時）:
    打開「speaker_1」資料夾，全選裡面的檔案，
    拖放到「Alice」資料夾中，
    再刪除空的「speaker_1」資料夾。

  重新命名或合併後，再次執行同一個音檔，
  就會得到使用正確講者名稱的結果。

  提示: 加上 --learning-mode 可為所有偵測到的講者
  （含已配對的）建立資料夾，方便手動檢視與修正。
------------------------------------------------------------
`, speakersDir)
}
