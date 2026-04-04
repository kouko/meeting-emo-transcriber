package commands

import (
	"fmt"
	"github.com/spf13/cobra"
)

func newTranscribeCmd() *cobra.Command {
	var (
		inputPath  string
		outputPath string
		format     string
		language   string
		threshold  float32
		noDiscover bool
	)
	cmd := &cobra.Command{
		Use:   "transcribe",
		Short: "Transcribe a meeting recording",
		RunE: func(cmd *cobra.Command, args []string) error {
			if inputPath == "" {
				return fmt.Errorf("--input is required")
			}
			fmt.Printf("Transcribe: input=%s speakers=%s format=%s language=%s\n", inputPath, speakersDir, format, language)
			fmt.Println("(not yet implemented — Phase 2)")
			return nil
		},
	}
	cmd.Flags().StringVar(&inputPath, "input", "", "input audio file path (required)")
	cmd.Flags().StringVar(&outputPath, "output", "", "output file path")
	cmd.Flags().StringVar(&format, "format", "txt", "output format: txt|json|srt|all (comma-separated)")
	cmd.Flags().StringVar(&language, "language", "auto", "language: auto|zh-TW|zh|en|ja")
	cmd.Flags().Float32Var(&threshold, "threshold", 0.6, "speaker similarity threshold")
	cmd.Flags().BoolVar(&noDiscover, "no-discover", false, "disable unknown speaker auto-discovery")
	cmd.MarkFlagRequired("input")
	return cmd
}
