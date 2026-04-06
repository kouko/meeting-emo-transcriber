package main

import (
	"fmt"
	"os"

	"github.com/kouko/meeting-emo-transcriber/cmd/commands"
)

// version is set via -ldflags "-X main.version=..." at build time.
var version = "dev"

func main() {
	if err := commands.NewRootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
