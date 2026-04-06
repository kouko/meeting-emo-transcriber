package commands

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/embedded"
	"github.com/spf13/cobra"
)

var (
	speakersDir string
	configPath  string
	logLevel    string
)

// audioExtensions that trigger implicit transcribe
var audioExtensions = map[string]bool{
	".wav": true, ".mp3": true, ".m4a": true, ".flac": true,
	".ogg": true, ".opus": true, ".aac": true, ".mp4": true,
	".mkv": true, ".webm": true,
}

func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "metr [audio file]",
		Short: "Meeting transcriber with speaker identification and emotion recognition",
	}
	root.PersistentFlags().StringVar(&speakersDir, "speakers", "./speakers", "speakers directory path")
	root.PersistentFlags().StringVar(&configPath, "config", "", "config file path (default: <speakers-dir>/_metr/config.yaml)")

	// Set speakers dir for portable mode detection (before any command runs)
	root.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		embedded.SetSpeakersDir(speakersDir)
	}
	root.PersistentFlags().StringVar(&logLevel, "log-level", "info", "log level: debug|info|warn|error")
	root.AddCommand(newInitCmd())
	root.AddCommand(newTranscribeCmd())
	root.AddCommand(newEnrollCmd())
	root.AddCommand(newSpeakersCmd())
	root.AddCommand(newPackCmd())
	root.AddCommand(newUnpackCmd())

	// Rewrite args: if first arg is an audio file, prepend "transcribe --input"
	originalArgs := os.Args[1:]
	if len(originalArgs) > 0 {
		first := originalArgs[0]
		// Skip if it's a known subcommand, flag, or help
		if first != "transcribe" && first != "enroll" && first != "speakers" &&
			first != "init" && first != "help" && first != "completion" &&
			first != "pack" && first != "unpack" &&
			!strings.HasPrefix(first, "-") {
			ext := strings.ToLower(filepath.Ext(first))
			if audioExtensions[ext] {
				if _, err := os.Stat(first); err == nil {
					newArgs := []string{"transcribe", "--input", first}
					newArgs = append(newArgs, originalArgs[1:]...)
					root.SetArgs(newArgs)
				}
			}
		}
	}

	return root
}
