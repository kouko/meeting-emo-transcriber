package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/spf13/cobra"
)

func newUnpackCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "unpack",
		Short: "Unpack speakers/_metr/ resources to ~/.metr/ for local use",
		RunE: func(cmd *cobra.Command, args []string) error {
			srcDir := filepath.Join(speakersDir, embedded.MetrDirName)
			dstDir := embedded.DefaultCacheDir()

			if _, err := os.Stat(srcDir); os.IsNotExist(err) {
				return fmt.Errorf("no portable resources found at %s", srcDir)
			}

			fmt.Printf("Unpacking %s → %s\n", srcDir, dstDir)

			dirs := []string{"bin", "models", "cache"}
			for _, d := range dirs {
				src := filepath.Join(srcDir, d)
				dst := filepath.Join(dstDir, d)
				if _, err := os.Stat(src); os.IsNotExist(err) {
					fmt.Printf("  ⏭ %s/ (not found, skipping)\n", d)
					continue
				}
				if err := copyDir(src, dst); err != nil {
					return fmt.Errorf("copy %s: %w", d, err)
				}
				size := dirSize(dst)
				fmt.Printf("  ✓ %s/ (%s)\n", d, formatSize(size))
			}

			totalSize := dirSize(dstDir)
			fmt.Printf("\nUnpacked to: %s (total %s)\n", dstDir, formatSize(totalSize))
			return nil
		},
	}
}
