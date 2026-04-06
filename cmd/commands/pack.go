package commands

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/spf13/cobra"
)

func newPackCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "pack",
		Short: "Pack binary, models, and cache into <speakers-dir>/_metr/ for portable use",
		RunE: func(cmd *cobra.Command, args []string) error {
			srcDir := embedded.DefaultCacheDir()
			dstDir := filepath.Join(speakersDir, embedded.MetrDirName)

			fmt.Printf("Packing %s → %s\n", srcDir, dstDir)

			// Copy bin/, models/, cache/
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

			// Also copy config.yaml if exists at root of srcDir
			srcConfig := filepath.Join(srcDir, "config.yaml")
			if _, err := os.Stat(srcConfig); err == nil {
				dstConfig := filepath.Join(dstDir, "config.yaml")
				copyFile(srcConfig, dstConfig)
				fmt.Printf("  ✓ config.yaml\n")
			}

			totalSize := dirSize(dstDir)
			fmt.Printf("\nPacked: %s (total %s)\n", dstDir, formatSize(totalSize))
			fmt.Printf("You can now copy %s/ to another Mac.\n", speakersDir)
			return nil
		},
	}
}

func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		dstPath := filepath.Join(dst, rel)

		if info.IsDir() {
			return os.MkdirAll(dstPath, 0755)
		}
		return copyFile(path, dstPath)
	})
}

func copyFile(src, dst string) error {
	os.MkdirAll(filepath.Dir(dst), 0755)
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}

	// Preserve executable permission
	info, _ := in.Stat()
	return os.Chmod(dst, info.Mode())
}

func dirSize(path string) int64 {
	var size int64
	filepath.Walk(path, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

func formatSize(bytes int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}
