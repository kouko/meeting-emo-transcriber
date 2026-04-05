package speaker

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// BatchEmbeddingFunc extracts speaker embeddings from multiple WAV files at once.
// Model is loaded once for all files. Returns embeddings parallel to input paths.
type BatchEmbeddingFunc func(wavPaths []string) ([][]float32, error)

// AutoEnroll checks all speakers in the store and recomputes embeddings
// for any that need updating. Uses batch extraction (model loaded once).
// Returns the number of speakers updated.
func AutoEnroll(store *Store, ffmpegPath string, batchExtractFn BatchEmbeddingFunc, forceAll ...bool) (int, error) {
	force := len(forceAll) > 0 && forceAll[0]
	names, err := store.List()
	if err != nil {
		return 0, err
	}

	updated := 0
	for _, name := range names {
		files, err := store.ListAudioFiles(name)
		if err != nil {
			return updated, err
		}
		if len(files) == 0 {
			continue
		}

		needsUpdate := force
		if !force {
			needsUpdate, err = store.NeedsUpdate(name)
			if err != nil {
				return updated, err
			}
		}
		if !needsUpdate {
			continue
		}

		fmt.Fprintf(os.Stderr, "  Auto-enrolling %s (%d samples)...\n", name, len(files))

		tmpDir, err := os.MkdirTemp("", "met-enroll-*")
		if err != nil {
			return updated, fmt.Errorf("create temp dir: %w", err)
		}

		// Convert all audio files to WAV first
		var wavPaths []string
		for i, file := range files {
			tempWav := filepath.Join(tmpDir, fmt.Sprintf("enroll_%d.wav", i))
			if err := audio.ConvertToWAV(ffmpegPath, file, tempWav); err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("convert %s: %w", file, err)
			}
			wavPaths = append(wavPaths, tempWav)
		}

		// Batch extract embeddings (model loaded once)
		embResults, err := batchExtractFn(wavPaths)
		if err != nil {
			os.RemoveAll(tmpDir)
			return updated, fmt.Errorf("batch extract embeddings for %s: %w", name, err)
		}

		var embeddings []types.SampleEmbedding
		for i, file := range files {
			if i >= len(embResults) {
				break
			}
			hash, err := FileHash(file)
			if err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("hash %s: %w", file, err)
			}
			embeddings = append(embeddings, types.SampleEmbedding{
				File:      filepath.Base(file),
				Hash:      hash,
				Embedding: embResults[i],
			})
		}
		os.RemoveAll(tmpDir)

		existing, _ := store.LoadProfile(name)
		now := time.Now().Format(time.RFC3339)
		dim := 0
		if len(embeddings) > 0 {
			dim = len(embeddings[0].Embedding)
		}
		profile := types.SpeakerProfile{
			Name:       name,
			Embeddings: embeddings,
			Dim:        dim,
			Model:      "wespeaker_v2",
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		if existing != nil && existing.CreatedAt != "" {
			profile.CreatedAt = existing.CreatedAt
		}

		if err := store.SaveProfile(profile); err != nil {
			return updated, fmt.Errorf("save profile %s: %w", name, err)
		}
		updated++
	}

	return updated, nil
}
