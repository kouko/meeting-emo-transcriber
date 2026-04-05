package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
	"github.com/spf13/cobra"
)

func newEnrollCmd() *cobra.Command {
	var force bool
	cmd := &cobra.Command{
		Use:   "enroll",
		Short: "Scan speakers/ directory and compute speaker embeddings",
		RunE: func(cmd *cobra.Command, args []string) error {
			store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
			names, err := store.List()
			if err != nil {
				return err
			}
			if len(names) == 0 {
				fmt.Printf("No speaker directories found in %s/\n", speakersDir)
				return nil
			}

			bins, err := embedded.ExtractAll()
			if err != nil {
				return fmt.Errorf("extract binaries: %w", err)
			}

			fmt.Printf("Scanning %s/...\n", speakersDir)

			var created, updated, unchanged int
			for _, name := range names {
				files, err := store.ListAudioFiles(name)
				if err != nil {
					return err
				}
				if len(files) == 0 {
					fmt.Printf("  %s: no audio files, skipping\n", name)
					continue
				}

				needsUpdate := force
				if !force {
					needsUpdate, err = store.NeedsUpdate(name)
					if err != nil {
						return err
					}
				}

				if !needsUpdate {
					fmt.Printf("  %s: %d samples → unchanged (cached)\n", name, len(files))
					unchanged++
					continue
				}

				var embeddings []types.SampleEmbedding
				tmpDir, err := os.MkdirTemp("", "met-enroll-*")
				if err != nil {
					return fmt.Errorf("create temp dir: %w", err)
				}

				for _, file := range files {
					tempWav := filepath.Join(tmpDir, "enroll.wav")
					if err := audio.ConvertToWAV(bins.FFmpeg, file, tempWav); err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("convert %s: %w", file, err)
					}

					// Extract embedding via FluidAudio WeSpeaker (256-dim)
					embResult, err := diarize.ExtractEmbedding(bins.Diarize, tempWav)
					if err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("extract embedding from %s: %w", file, err)
					}

					emb32 := make([]float32, len(embResult.Embedding))
					for i, v := range embResult.Embedding {
						emb32[i] = float32(v)
					}

					hash, err := speaker.FileHash(file)
					if err != nil {
						os.RemoveAll(tmpDir)
						return fmt.Errorf("hash %s: %w", file, err)
					}

					embeddings = append(embeddings, types.SampleEmbedding{
						File:      filepath.Base(file),
						Hash:      hash,
						Embedding: emb32,
					})
				}
				os.RemoveAll(tmpDir)

				status := "created"
				existing, _ := store.LoadProfile(name)
				if existing != nil {
					status = "updated"
				}

				now := time.Now().Format(time.RFC3339)
				dim := 0
				if len(embeddings) > 0 {
					dim = len(embeddings[0].Embedding)
				}
				profile := types.SpeakerProfile{
					Name:       name,
					Embeddings: embeddings,
					Dim:        dim,
					Model:      "wespeaker_v2",
					CreatedAt:  now,
					UpdatedAt:  now,
				}
				if existing != nil && existing.CreatedAt != "" {
					profile.CreatedAt = existing.CreatedAt
				}

				if err := store.SaveProfile(profile); err != nil {
					return fmt.Errorf("save profile %s: %w", name, err)
				}

				fmt.Printf("  %s: %d samples → embedding computed ✓ (%s)\n", name, len(files), status)
				if status == "created" {
					created++
				} else {
					updated++
				}
			}

			fmt.Printf("\n%d created, %d updated, %d unchanged.\n", created, updated, unchanged)
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "force recompute all embeddings")
	return cmd
}
