package main

import (
	"bytes"
	"os/exec"
	"testing"
)

// TestMainBinaryHasNoSherpaDep enforces the single most important
// invariant of the sherpa-sidecar refactor: the main metr binary must
// never transitively depend on sherpa-onnx-go-macos.
//
// If this test fails, it means somebody added an import that reintroduces
// the cgo sherpa dependency into the main binary. When that happens:
//
//   - `go build ./cmd/` will once again produce a Mach-O with a baked-in
//     LC_RPATH pointing at /Users/<builder>/go/pkg/mod/...
//   - Released binaries will fail on end-user machines with
//     `dyld: Library not loaded: @rpath/libsherpa-onnx-c-api.dylib`
//   - The single-file portability property is gone.
//
// All sherpa usage must stay in `cmd/metr-sherpa` and its exclusive
// support package `internal/sherpasidecar/sherpasrv`. The main binary
// talks to that sidecar over gob-encoded stdin/stdout via
// `internal/sherpasidecar` (pure Go, no cgo).
//
// The test runs `go list -deps ./cmd/` and searches the output. We use
// `-deps` rather than building the binary because `go list` is fast (~1s)
// and doesn't require CGO or the embedded FS to be populated.
func TestMainBinaryHasNoSherpaDep(t *testing.T) {
	// CWD when `go test` runs is the package dir (./cmd), so "." here
	// means the main metr package itself.
	out, err := exec.Command("go", "list", "-deps", ".").Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			t.Fatalf("go list -deps failed: %v\n%s", err, ee.Stderr)
		}
		t.Fatalf("go list -deps failed: %v", err)
	}
	banned := []string{
		"github.com/k2-fsa/sherpa-onnx-go-macos",
		"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar/sherpasrv",
	}
	for _, dep := range banned {
		if bytes.Contains(out, []byte(dep)) {
			t.Errorf(`main binary (./cmd/) transitively depends on %q.

This violates the sherpa-sidecar invariant. All sherpa-onnx usage must
live in ./cmd/metr-sherpa and ./internal/sherpasidecar/sherpasrv only.
The main metr binary must stay cgo-free so it can be shipped as a
single-file Mach-O.

If the new import is intentional, the refactor has been undone and
Homebrew users will see "dyld: Library not loaded" on first run. Revert
the offending import or move the consumer into cmd/metr-sherpa.`, dep)
		}
	}
}
