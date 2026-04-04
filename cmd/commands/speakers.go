package commands

import (
	"fmt"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
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
	var name, audio string
	cmd := &cobra.Command{
		Use:   "verify",
		Short: "Verify speaker recognition accuracy",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Printf("Verify: name=%s audio=%s\n", name, audio)
			fmt.Println("(not yet implemented — Phase 3)")
			return nil
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "speaker name (required)")
	cmd.Flags().StringVar(&audio, "audio", "", "test audio file path (required)")
	cmd.MarkFlagRequired("name")
	cmd.MarkFlagRequired("audio")
	return cmd
}
