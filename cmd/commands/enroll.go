package commands

import (
	"fmt"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
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

			// Extract voiceprint from concatenated wav
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

			fmt.Printf("Scanning %s/...\n", speakersDir)

			enrolled, err := speaker.AutoEnroll(store, bins.FFmpeg, extractFn, force)
			if err != nil {
				return fmt.Errorf("enroll: %w", err)
			}

			unchanged := len(names) - enrolled
			fmt.Printf("\n%d enrolled, %d unchanged.\n", enrolled, unchanged)
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "force recompute all embeddings")
	return cmd
}
