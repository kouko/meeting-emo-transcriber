BINARY_NAME := metr
GOOS ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')
GOARCH ?= $(shell uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS := -X main.version=$(VERSION)

.PHONY: all build clean deps build-deps download-deps test info clean-all

# Default: build for current platform
all: deps build

# Download/build all external dependencies
deps: build-deps download-deps

build-deps:
	@echo "==> Building whisper-cli..."
	@bash scripts/build-whisper.sh
	@echo "==> Building metr-diarize..."
	@bash scripts/build-diarize.sh

download-deps:
	@echo "==> Downloading ffmpeg..."
	@bash scripts/download-ffmpeg.sh

# Build the Go binary
build:
	@echo "==> Building $(BINARY_NAME) $(VERSION) for $(GOOS)-$(GOARCH)..."
	go build -ldflags "$(LDFLAGS)" -o $(BINARY_NAME) ./cmd/

# Run tests
test:
	go test ./... -v

# Clean build artifacts (keep deps)
clean:
	rm -f $(BINARY_NAME) main
	rm -rf dist/

# Clean everything including deps
clean-all: clean
	rm -rf embedded/bin/darwin-arm64/
	rm -rf .build/
	rm -rf tools/metr-diarize/.build/

# Show current build info
info:
	@echo "GOOS=$(GOOS) GOARCH=$(GOARCH) VERSION=$(VERSION)"
	@echo "Deps dir: embedded/bin/darwin-arm64/"
	@ls -lh embedded/bin/darwin-arm64/ 2>/dev/null || echo "(no deps yet — run 'make deps')"
