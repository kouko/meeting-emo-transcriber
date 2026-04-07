BINARY_NAME := metr
GOOS ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')
GOARCH ?= $(shell uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -X main.version=$(VERSION)

.PHONY: all build clean deps build-deps download-deps build-sherpa-sidecar audit-embedded test info clean-all package

# Default: build for current platform
all: deps audit-embedded build

# Download/build all external dependencies
deps: build-deps download-deps build-sherpa-sidecar

# Audit every binary under embedded/bin/<platform>/ for LC_RPATH entries
# and @rpath dependencies that would fail on end-user machines. This
# catches both the sherpa-onnx and whisper-cli classes of dyld bugs
# before they reach a tagged release.
audit-embedded:
	@echo "==> Auditing embedded binaries for dyld load problems..."
	@bash scripts/audit-embedded-binaries.sh

build-deps:
	@echo "==> Building whisper-cli..."
	@bash scripts/build-whisper.sh
	@echo "==> Building metr-diarize..."
	@bash scripts/build-diarize.sh
	@echo "==> Building metr-denoise..."
	@bash scripts/build-denoise.sh

# Build the metr-sherpa sidecar + copy sherpa-onnx dylibs into
# embedded/bin/ so go:embed picks them up. Kept separate from build-deps
# because it depends on Go module cache being populated (go.sum), which
# happens on first `go build`, whereas build-deps runs Swift/C++ builds
# that don't touch Go modules.
build-sherpa-sidecar:
	@echo "==> Building metr-sherpa sidecar..."
	@bash scripts/build-sherpa-sidecar.sh

download-deps:
	@echo "==> Building ffmpeg (LGPL, from source)..."
	@bash scripts/build-ffmpeg.sh

# Build the Go binary
build:
	@echo "==> Building $(BINARY_NAME) $(VERSION) for $(GOOS)-$(GOARCH)..."
	go build -ldflags "$(LDFLAGS)" -o $(BINARY_NAME) ./cmd/

# Run tests
test:
	go test ./... -v

# Package binary as tarball for release
package: build
	@mkdir -p dist
	tar czf dist/$(BINARY_NAME)-$(GOOS)-$(GOARCH).tar.gz $(BINARY_NAME)
	@echo "==> Packaged: dist/$(BINARY_NAME)-$(GOOS)-$(GOARCH).tar.gz"

# Clean build artifacts (keep deps)
clean:
	rm -f $(BINARY_NAME) main
	rm -rf dist/

# Clean everything including deps (sidecar binary, dylibs, swift/cpp builds)
clean-all: clean
	rm -rf embedded/bin/darwin-arm64/
	rm -rf .build/
	rm -rf tools/metr-diarize/.build/
	rm -rf tools/metr-denoise/.build/

# Show current build info
info:
	@echo "GOOS=$(GOOS) GOARCH=$(GOARCH) VERSION=$(VERSION)"
	@echo "Deps dir: embedded/bin/darwin-arm64/"
	@ls -lh embedded/bin/darwin-arm64/ 2>/dev/null || echo "(no deps yet — run 'make deps')"
