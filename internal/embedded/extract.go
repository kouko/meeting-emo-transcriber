package embedded

import (
	"fmt"
	"os"
	"path/filepath"
)

// Data provider function variables. In dev mode these return placeholder bytes.
// embed_prod.go overrides them via init() when built with the "embed" build tag.
var whisperCLIData = func() []byte { return []byte("whisper-cli-placeholder") }
var ffmpegData = func() []byte { return []byte("ffmpeg-placeholder") }
var onnxRuntimeData = func() []byte { return []byte("onnxruntime-placeholder") }

// binSpec describes a single binary to be extracted.
type binSpec struct {
	name     string        // file name inside bin/
	dataFunc func() []byte // data provider
}

// ExtractAll ensures all bundled binaries are extracted to CacheDir()/bin/,
// skipping any whose SHA-256 hash matches the cached record in .versions.json.
// Returns BinPaths with absolute paths to each binary.
func ExtractAll() (BinPaths, error) {
	cacheDir := CacheDir()
	binDir := filepath.Join(cacheDir, "bin")

	if err := os.MkdirAll(binDir, 0755); err != nil {
		return BinPaths{}, fmt.Errorf("create bin dir: %w", err)
	}

	versionsPath := filepath.Join(cacheDir, versionsFile)
	versions, err := loadVersions(versionsPath)
	if err != nil {
		return BinPaths{}, fmt.Errorf("load versions: %w", err)
	}

	specs := []binSpec{
		{name: "whisper-cli", dataFunc: whisperCLIData},
		{name: "ffmpeg", dataFunc: ffmpegData},
		{name: "libonnxruntime.dylib", dataFunc: onnxRuntimeData},
	}

	paths := make(map[string]string, len(specs))
	for _, spec := range specs {
		data := spec.dataFunc()
		hash, err := computeSHA256(data)
		if err != nil {
			return BinPaths{}, fmt.Errorf("sha256 %s: %w", spec.name, err)
		}
		destPath := filepath.Join(binDir, spec.name)
		if _, err := extractBinary(data, destPath, hash, versions); err != nil {
			return BinPaths{}, fmt.Errorf("extract %s: %w", spec.name, err)
		}
		paths[spec.name] = destPath
	}

	if err := saveVersions(versionsPath, versions); err != nil {
		return BinPaths{}, fmt.Errorf("save versions: %w", err)
	}

	return BinPaths{
		WhisperCLI:  paths["whisper-cli"],
		FFmpeg:      paths["ffmpeg"],
		ONNXRuntime: paths["libonnxruntime.dylib"],
	}, nil
}
