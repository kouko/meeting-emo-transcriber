package commands

import (
	"fmt"
	"os"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/diarize"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/spf13/cobra"
)

func newEnrollCmd() *cobra.Command {
	var (
		force       bool
		rebuild     bool
		rebuildName string
	)
	cmd := &cobra.Command{
		Use:   "enroll",
		Short: "Scan speakers directory and compute speaker embeddings",
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

			if rebuild {
				targets := names
				if rebuildName != "" {
					found := false
					for _, n := range names {
						if n == rebuildName {
							found = true
							break
						}
					}
					if !found {
						return fmt.Errorf("speaker %q not found in %s", rebuildName, speakersDir)
					}
					targets = []string{rebuildName}
				}
				fmt.Printf("Rebuilding %d speaker(s) in %s/ (existing profiles will be backed up to *.bak.<ts>.json)...\n",
					len(targets), speakersDir)
				var rebuilt, failed int
				for _, name := range targets {
					fmt.Fprintf(os.Stderr, "  Rebuilding %s...\n", name)
					if err := speaker.RebuildSpeaker(store, name, bins.FFmpeg, extractFn); err != nil {
						fmt.Fprintf(os.Stderr, "  ⚠ rebuild %s failed: %v\n", name, err)
						failed++
						continue
					}
					rebuilt++
				}
				fmt.Printf("\n%d rebuilt, %d failed.\n", rebuilt, failed)
				if failed > 0 {
					return fmt.Errorf("%d speaker(s) failed to rebuild", failed)
				}
				return nil
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
	cmd.Flags().BoolVar(&force, "force", false, "force recompute merged voiceprint for speakers (keeps existing centroids)")
	cmd.Flags().BoolVar(&rebuild, "rebuild", false, "discard ALL stored voiceprints and re-extract from scratch under the current per-file averaging recipe (backs up first; pair with --speaker to scope)")
	cmd.Flags().StringVar(&rebuildName, "speaker", "", "limit --rebuild to a single speaker; ignored without --rebuild")
	return cmd
}
