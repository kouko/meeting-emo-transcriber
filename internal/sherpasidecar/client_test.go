package sherpasidecar

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestClientLifecycleWithRealSidecar spawns the real metr-sherpa binary
// and exercises the lifecycle that doesn't need models: Spawn → Close.
// It also verifies that calling methods after Close fails fast.
//
// The test `go build`s the sidecar into a temp dir, which keeps it
// hermetic and doesn't rely on `make deps`. It skips gracefully if
// `go build` is unavailable (e.g. minimal CI images) or the build fails
// for reasons unrelated to this package.
func TestClientLifecycleWithRealSidecar(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}

	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("go toolchain not found: %v", err)
	}

	tmp := t.TempDir()
	binPath := filepath.Join(tmp, "metr-sherpa")

	// Build the sidecar. We use "./..." rooted at the repo via a relative
	// path — since `go test` runs with CWD at the package being tested,
	// we need to walk up to repo root.
	repoRoot, err := findRepoRoot()
	if err != nil {
		t.Fatalf("findRepoRoot: %v", err)
	}

	build := exec.Command(goBin, "build", "-o", binPath, "./cmd/metr-sherpa")
	build.Dir = repoRoot
	build.Env = append(os.Environ(), "CGO_ENABLED=1")
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("could not build metr-sherpa (likely missing sherpa-onnx deps in test env): %v\n%s", err, out)
	}

	client, err := Spawn(binPath)
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}

	// A Punctuate call without LoadPunctuator should return a clean error,
	// not hang or crash the sidecar.
	if _, err := client.Punctuate("hello", "en"); err == nil {
		t.Error("expected error from Punctuate before LoadPunctuator")
	}

	if err := client.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Second Close is a no-op.
	if err := client.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}

	// After Close, calls fail fast (no hang).
	if _, err := client.Punctuate("hello", "en"); err == nil {
		t.Error("expected error from Punctuate after Close")
	}
}

// findRepoRoot walks up from the current working directory looking for a
// go.mod file. Used by tests that need to invoke the Go toolchain against
// the whole module.
func findRepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}
