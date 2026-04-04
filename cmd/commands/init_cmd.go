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
				template := "# speakers/config.yaml\n# Uncomment and modify values as needed.\n\n# language: \"auto\"        # auto | zh-TW | zh | en | ja\n# threshold: 0.6          # Speaker similarity threshold\n# format: txt             # txt | json | srt | all\n# discover: true          # Auto-discover unknown speakers\n"
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
