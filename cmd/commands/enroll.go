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

			// Batch extract function (model loaded once per speaker)
			batchExtractFn := func(wavPaths []string) ([][]float32, error) {
				results, err := diarize.ExtractVoiceprints(bins.Diarize, wavPaths)
				if err != nil {
					return nil, err
				}
				out := make([][]float32, len(results))
				for i, r := range results {
					emb := make([]float32, len(r.Vector))
					for j, v := range r.Vector {
						emb[j] = float32(v)
					}
					out[i] = emb
				}
				return out, nil
			}

			fmt.Printf("Scanning %s/...\n", speakersDir)

			enrolled, err := speaker.AutoEnroll(store, bins.FFmpeg, batchExtractFn, force)
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
