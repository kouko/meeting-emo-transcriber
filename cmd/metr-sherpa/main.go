// Command metr-sherpa is a sidecar binary that owns all sherpa-onnx-go-macos
// usage in the metr toolchain. It is spawned by the main metr process,
// communicates over stdin/stdout via encoding/gob, and exits when its stdin
// closes or it receives a shutdown envelope.
//
// This exists so the main metr binary can stay cgo-free: dyld resolves a
// process's own dylib dependencies before main() runs, so if the main
// binary itself linked sherpa-onnx it could never be truly relocatable. By
// moving the cgo boundary into a sidecar that's extracted to the runtime
// cache dir alongside its dylibs, the main binary has nothing to resolve
// at startup and can be shipped as a single file.
package main

import (
	"fmt"
	"os"

	"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar/sherpasrv"
)

func main() {
	// Serve blocks until stdin closes or a MsgShutdown arrives. Any error
	// here is a protocol-level failure (malformed gob stream, pipe broken
	// while reading) and we surface it on stderr so the parent can see it
	// in its own logs. stdout is reserved for the gob reply stream.
	if err := sherpasrv.Serve(os.Stdin, os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "metr-sherpa: %v\n", err)
		os.Exit(1)
	}
}
