package commands

import "github.com/spf13/cobra"

var (
	speakersDir string
	configPath  string
	logLevel    string
)

func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "meeting-emo-transcriber",
		Short: "Meeting transcriber with speaker identification and emotion recognition",
	}
	root.PersistentFlags().StringVar(&speakersDir, "speakers", "./speakers", "speakers directory path")
	root.PersistentFlags().StringVar(&configPath, "config", "", "config file path (default: <speakers-dir>/config.yaml)")
	root.PersistentFlags().StringVar(&logLevel, "log-level", "info", "log level: debug|info|warn|error")
	root.AddCommand(newInitCmd())
	root.AddCommand(newTranscribeCmd())
	root.AddCommand(newEnrollCmd())
	root.AddCommand(newSpeakersCmd())
	return root
}
