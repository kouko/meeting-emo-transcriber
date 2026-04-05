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
				needsUpdate, _ := store.NeedsUpdate(name)
				status := "enrolled"
				if needsUpdate {
					status = "needs enrollment"
				}
				fmt.Printf("  %s: %d samples (%s)\n", name, len(files), status)
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
			if len(profile.Embeddings) == 0 {
				return fmt.Errorf("speaker %q has no embeddings (run enroll first)", name)
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

			// Extract embedding via FluidAudio WeSpeaker (256-dim)
			embResult, err := diarize.ExtractEmbedding(bins.Diarize, tempWav)
			if err != nil {
				return fmt.Errorf("extract embedding: %w", err)
			}

			testEmb := make([]float32, len(embResult.Embedding))
			for i, v := range embResult.Embedding {
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
