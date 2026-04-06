package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

func newInitCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Initialize speakers directory and config template",
		RunE: func(cmd *cobra.Command, args []string) error {
			metrDir := filepath.Join(speakersDir, "_metr")
			dirs := []string{speakersDir, metrDir}
			for _, d := range dirs {
				if err := os.MkdirAll(d, 0755); err != nil {
					return fmt.Errorf("creating %s: %w", d, err)
				}
				fmt.Printf("  Created %s/\n", d)
			}
			cfgPath := filepath.Join(metrDir, "config.yaml")
			if _, err := os.Stat(cfgPath); os.IsNotExist(err) {
				template := "# metr config.yaml\n# Uncomment and modify values as needed.\n\n# language: \"auto\"        # auto | zh-TW | zh | en | ja\n# threshold: 0.8          # Diarization clustering threshold (higher = more speakers)\n# match_threshold: 0.55   # Speaker matching threshold for enrolled profiles\n# format: txt             # txt | json | srt | all\n\n# Custom vocabulary for better ASR accuracy (names, terms, jargon)\n# vocabulary:\n#   - \"kouko\"\n#   - \"YanJen\"\n"
				if err := os.WriteFile(cfgPath, []byte(template), 0644); err != nil {
					return fmt.Errorf("writing config template: %w", err)
				}
				fmt.Printf("  Created %s\n", cfgPath)
			}
			fmt.Printf("\nWorkspace initialized at %s\n", speakersDir)
			fmt.Println("Add speaker samples to <speakers-dir>/<name>/ and run 'metr <audio file>'.")
			return nil
		},
	}
}
