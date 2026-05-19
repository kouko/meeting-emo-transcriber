package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newSpeakersCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "speakers",
		Short: "Speaker management commands",
	}
	cmd.AddCommand(newSpeakersListCmd())
	cmd.AddCommand(newSpeakersVerifyCmd())
	cmd.AddCommand(newSpeakersInspectCmd())
	return cmd
}

func newSpeakersListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all registered speakers",
		RunE: func(cmd *cobra.Command, args []string) error {
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			names, err := store.List()
			if err != nil {
				return err
			}
			if len(names) == 0 {
				fmt.Println("No speakers found.")
				return nil
			}
			for _, name := range names {
				files, _ := store.ListAudioFiles(name)
				profile, _ := store.LoadProfile(name)
				status := "no profile"
				if profile != nil && len(profile.Voiceprints) > 0 {
					newFiles, _ := store.FindNewAudioFiles(name)
					if len(newFiles) > 0 {
						status = fmt.Sprintf("enrolled, %d new samples", len(newFiles))
					} else {
						status = fmt.Sprintf("enrolled, %d voiceprints", len(profile.Voiceprints))
					}
				}
				fmt.Printf("  %s: %d audio files (%s)\n", name, len(files), status)
			}
			return nil
		},
	}
}

func newSpeakersVerifyCmd() *cobra.Command {
	var name, audioPath string
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify speaker recognition accuracy",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(configPath, speakersDir)
			if err != nil {
				return fmt.Errorf("load config: %w", err)
			}

			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			profile, err := store.LoadProfile(name)
			if err != nil {
				return fmt.Errorf("load profile: %w", err)
			}
			if profile == nil {
				return fmt.Errorf("speaker %q not found in %s", name, speakersDir)
			}
			if len(profile.Voiceprints) == 0 {
				return fmt.Errorf("speaker %q has no voiceprints (run enroll first)", name)
			}

			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			// Convert test audio to WAV
			tmpDir, err := os.MkdirTemp("", "met-verify-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			tempWav := filepath.Join(tmpDir, "verify.wav")
			if err := audio.ConvertToWAV(bins.FFmpeg, audioPath, tempWav); err != nil {
				return fmt.Errorf("convert audio: %w", err)
			}

			// Extract voiceprint via FluidAudio WeSpeaker (256-dim)
			vpResult, err := diarize.ExtractVoiceprint(bins.Diarize, tempWav)
			if err != nil {
				return fmt.Errorf("extract voiceprint: %w", err)
			}

			testEmb := make([]float32, len(vpResult.Vector))
			for i, v := range vpResult.Vector {
				testEmb[i] = float32(v)
			}

			// Match against enrolled profile
			matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})
			result := matcher.Match(testEmb, []types.SpeakerProfile{*profile}, float32(cfg.Threshold))

			fmt.Printf("Verifying against %s...\n", name)
			fmt.Printf("  Similarity: %.2f (threshold: %.2f)\n", result.Similarity, cfg.Threshold)
			if result.Name != "" {
				fmt.Printf("  Result: ✓ MATCH\n")
			} else {
				fmt.Printf("  Result: ✗ NO MATCH\n")
			}

			return nil
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "speaker name (required)")
	cmd.Flags().StringVar(&audioPath, "audio", "", "test audio file path (required)")
	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("audio")
	return cmd
}

func newSpeakersInspectCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "inspect <speaker-name>",
		Short: "Inspect an enrolled speaker's voiceprint quality (intra/inter cosine)",
		Long: `Inspect re-extracts an embedding from every audio file enrolled for
a speaker, then reports:

  - Per-file cosine similarity to the speaker's merged voiceprint
    (intra-class — high & consistent values mean reliable enrollment).
  - Highest cosine vs every other enrolled speaker
    (inter-class — high values flag a confusion risk).
  - A "safety margin" = intra-min minus inter-max. Positive ≈ this
    speaker is well separated from everyone else in the set.

Note: intra-class and safety margin are computed against the single
"merged" voiceprint stored in the profile. The live matcher used during
transcribe scores each cluster against every stored voiceprint and
keeps the max (BestSimilarity), so it is meaningfully more forgiving
than the worst-case margin reported here. A negative margin is a
warning, not a guarantee of misidentification.

Useful for tuning --match-threshold and --match-margin without running
a full transcribe.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			target := args[0]
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())

			targetProfile, err := store.LoadProfile(target)
			if err != nil {
				return fmt.Errorf("load profile for %s: %w", target, err)
			}
			if targetProfile == nil {
				return fmt.Errorf("speaker %q not found under %s", target, speakersDir)
			}
			targetProfile.Name = target

			merged := speaker.MergedVoiceprint(targetProfile)
			if merged == nil {
				return fmt.Errorf("speaker %q has no merged voiceprint — run `metr enroll` first", target)
			}

			audioFiles, err := store.ListAudioFiles(target)
			if err != nil {
				return fmt.Errorf("list audio files: %w", err)
			}
			if len(audioFiles) == 0 {
				return fmt.Errorf("speaker %q has no audio samples", target)
			}

			otherNames, err := store.List()
			if err != nil {
				return fmt.Errorf("list speakers: %w", err)
			}
			var otherProfiles []types.SpeakerProfile
			for _, n := range otherNames {
				if n == target {
					continue
				}
				p, err := store.LoadProfile(n)
				if err != nil || p == nil {
					continue
				}
				p.Name = n
				otherProfiles = append(otherProfiles, *p)
			}

			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			tmpDir, err := os.MkdirTemp("", "met-inspect-*")
			if err != nil {
				return fmt.Errorf("create temp dir: %w", err)
			}
			defer os.RemoveAll(tmpDir)

			// Convert + extract each audio file
			wavPaths := make([]string, 0, len(audioFiles))
			fileDurations := make([]float64, 0, len(audioFiles))
			fileNames := make([]string, 0, len(audioFiles))
			var totalDuration float64
			for i, src := range audioFiles {
				tempWav := filepath.Join(tmpDir, fmt.Sprintf("inspect_%d.wav", i))
				if err := audio.ConvertToWAV(bins.FFmpeg, src, tempWav); err != nil {
					return fmt.Errorf("convert %s: %w", src, err)
				}
				samples, rate, err := audio.ReadWAV(tempWav)
				if err != nil {
					fmt.Fprintf(os.Stderr, "warning: cannot read %s: %v\n", tempWav, err)
					continue
				}
				dur := float64(len(samples)) / float64(rate)
				wavPaths = append(wavPaths, tempWav)
				fileDurations = append(fileDurations, dur)
				fileNames = append(fileNames, filepath.Base(src))
				totalDuration += dur
			}

			if len(wavPaths) == 0 {
				return fmt.Errorf("no readable audio files for %s", target)
			}

			vpResults, err := diarize.ExtractVoiceprints(bins.Diarize, wavPaths)
			if err != nil {
				return fmt.Errorf("extract voiceprints: %w", err)
			}
			embeddings := make([][]float32, len(vpResults))
			for i, vp := range vpResults {
				emb := make([]float32, len(vp.Vector))
				for j, v := range vp.Vector {
					emb[j] = float32(v)
				}
				embeddings[i] = emb
			}

			report := speaker.ComputeInspection(target, embeddings, merged, otherProfiles)

			fmt.Printf("Inspecting speaker %q...\n\n", target)
			fmt.Printf("Audio samples:\n")
			for i, name := range fileNames {
				fmt.Printf("  %-30s %5.1fs\n", name, fileDurations[i])
			}
			fmt.Printf("  %-30s %5.1fs", "Total:", totalDuration)
			if totalDuration >= speaker.EnrollMinDurationSec {
				fmt.Printf(" ✓ (>= %.0fs minimum)\n\n", speaker.EnrollMinDurationSec)
			} else {
				fmt.Printf(" ⚠ (below %.0fs minimum)\n\n", speaker.EnrollMinDurationSec)
			}

			fmt.Printf("Per-file vs merged voiceprint (intra-class):\n")
			for i, sim := range report.IntraSims {
				marker := " "
				if sim < 0.5 {
					marker = "⚠"
				}
				fmt.Printf("  %s %-30s sim %.2f\n", marker, fileNames[i], sim)
			}
			fmt.Printf("  ─ mean %.2f  min %.2f  max %.2f\n\n",
				report.IntraMean, report.IntraMin, report.IntraMax)

			if len(report.Inter) > 0 {
				fmt.Printf("vs Other enrolled profiles (inter-class):\n")
				for _, s := range report.Inter {
					marker := " "
					if s.MaxSim > 0.5 {
						marker = "⚠"
					}
					fmt.Printf("  %s %-30s max sim %.2f\n", marker, s.OtherName, s.MaxSim)
				}
				fmt.Println()
				fmt.Printf("Safety margin: %.2f (intra-min %.2f − strongest impostor %.2f)\n",
					report.SafetyMargin, report.IntraMin, report.Inter[0].MaxSim)
				fmt.Printf("  (computed vs the merged voiceprint only; live matcher uses max over all stored voiceprints and is more forgiving)\n")
				if report.SafetyMargin < 0 {
					fmt.Printf("  ⚠ negative margin against the merged voiceprint — at least one other speaker's stored voiceprint scores higher than this speaker's worst sample\n")
				} else if report.SafetyMargin < 0.1 {
					fmt.Printf("  ⚠ tight margin — consider raising --match-threshold or adding more enrollment audio\n")
				}
			} else {
				fmt.Printf("(no other enrolled speakers to compare against)\n")
			}

			return nil
		},
	}
}
