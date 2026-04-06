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
		Short: "Initialize working directory (creates speakers/ + output/ + config.yaml template)",
		RunE: func(cmd *cobra.Command, args []string) error {
			dirs := []string{"speakers", "output"}
			for _, d := range dirs {
				if err := os.MkdirAll(d, 0755); err != nil {
					return fmt.Errorf("creating %s: %w", d, err)
				}
				fmt.Printf("  Created %s/\n", d)
			}
			configPath := filepath.Join("speakers", "config.yaml")
			if _, err := os.Stat(configPath); os.IsNotExist(err) {
				template := "# speakers/config.yaml\n# Uncomment and modify values as needed.\n\n# language: \"auto\"        # auto | zh-TW | zh | en | ja\n# threshold: 0.8          # Diarization clustering threshold (higher = more speakers)\n# format: txt             # txt | json | srt | all\n\n# Custom vocabulary for better ASR accuracy (names, terms, jargon)\n# vocabulary:\n#   - \"kouko\"\n#   - \"YanJen\"\n#   - \"Claude\"\n#   - \"Kubernetes\"\n"
				if err := os.WriteFile(configPath, []byte(template), 0644); err != nil {
					return fmt.Errorf("writing config template: %w", err)
				}
				fmt.Printf("  Created %s\n", configPath)
			}
			fmt.Println("\nWorkspace initialized. Add speaker samples to speakers/<name>/ and run 'transcribe'.")
			return nil
		},
	}
}
