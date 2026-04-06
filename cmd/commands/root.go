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

// defaultSpeakersDir returns the default speakers directory.
// Priority: ./metr-speakers (portable, if exists) > ~/metr-speakers (global)
func defaultSpeakersDir() string {
	local := "./metr-speakers"
	if info, err := os.Stat(local); err == nil && info.IsDir() {
		return local
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return local
	}
	return filepath.Join(home, "metr-speakers")
}

func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "metr [audio file]",
		Short: "Meeting transcriber with speaker identification and emotion recognition",
		Long: `metr - Meeting transcriber with speaker identification and emotion recognition

Quick start:
  metr meeting.mp3                            Transcribe with defaults
  metr meeting.mp3 --language ja              Specify language
  metr meeting.mp3 --format all               Output txt + json + srt
  metr meeting.mp3 --enhance                  Enable noise reduction (DeepFilterNet3)
  metr meeting.mp3 --normalize                Force loudnorm normalization
  metr meeting.mp3 --prompt "Alice, ACME"     Custom vocabulary hints for ASR
  metr meeting.mp3 --threshold 0.6            Fewer speakers (lower = more merging)
  metr meeting.mp3 --match-threshold 0.7      Stricter speaker matching
  metr meeting.mp3 --learning-mode            Create folders for all detected speakers
  metr meeting.mp3 --no-cache                 Force re-transcription (skip cache)

Speakers directory:
  Default: ~/metr-speakers (global, shared across all directories)
  If ./metr-speakers/ exists, uses it instead (portable mode)
  Override: --speakers /path/to/dir

Subcommands:
  metr transcribe --input meeting.mp3       Same as above (explicit form)
  metr enroll                               Enroll speaker voiceprints
  metr speakers                             List enrolled speakers
  metr pack / unpack                        Portable mode (metr-speakers/_metr/)`,
	}
	root.PersistentFlags().StringVar(&speakersDir, "speakers", "", "speakers directory path (default: ~/metr-speakers)")
	root.PersistentFlags().StringVar(&configPath, "config", "", "config file path (default: <speakers-dir>/_metr/config.yaml)")

	// Resolve speakers dir: explicit flag > ./metr-speakers (portable) > ~/metr-speakers (global)
	root.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		if speakersDir == "" {
			speakersDir = defaultSpeakersDir()
		}
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
