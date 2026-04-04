package commands

import (
	"fmt"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/spf13/cobra"
)

func newEnrollCmd() *cobra.Command {
	var force bool
	cmd := &cobra.Command{
		Use:   "enroll",
		Short: "Scan speakers/ directory and register all speakers",
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
			fmt.Printf("Scanning %s/...\n", speakersDir)
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
				if needsUpdate {
					fmt.Printf("  %s: %d samples → embedding computation pending (Phase 3)\n", name, len(files))
				} else {
					fmt.Printf("  %s: %d samples → unchanged (cached)\n", name, len(files))
				}
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "force recompute all embeddings")
	return cmd
}
