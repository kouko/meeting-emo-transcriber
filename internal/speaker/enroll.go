package speaker

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// AutoEnroll checks all speakers in the store and recomputes embeddings
// for any that need updating. Returns the number of speakers updated.
// ffmpegPath is needed to convert audio files to WAV.
func AutoEnroll(store *Store, extractor *Extractor, ffmpegPath string) (int, error) {
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

		needsUpdate, err := store.NeedsUpdate(name)
		if err != nil {
			return updated, err
		}
		if !needsUpdate {
			continue
		}

		fmt.Fprintf(os.Stderr, "  Auto-enrolling %s (%d samples)...\n", name, len(files))

		var embeddings []types.SampleEmbedding
		tmpDir, err := os.MkdirTemp("", "met-enroll-*")
		if err != nil {
			return updated, fmt.Errorf("create temp dir: %w", err)
		}

		for _, file := range files {
			tempWav := filepath.Join(tmpDir, "enroll.wav")
			if err := audio.ConvertToWAV(ffmpegPath, file, tempWav); err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("convert %s: %w", file, err)
			}

			samples, sampleRate, err := audio.ReadWAV(tempWav)
			if err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("read %s: %w", file, err)
			}

			emb, err := extractor.Extract(samples, sampleRate)
			if err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("extract embedding from %s: %w", file, err)
			}

			hash, err := FileHash(file)
			if err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("hash %s: %w", file, err)
			}

			embeddings = append(embeddings, types.SampleEmbedding{
				File:      filepath.Base(file),
				Hash:      hash,
				Embedding: emb,
			})
		}
		os.RemoveAll(tmpDir)

		existing, _ := store.LoadProfile(name)
		now := time.Now().Format(time.RFC3339)
		profile := types.SpeakerProfile{
			Name:       name,
			Embeddings: embeddings,
			Dim:        extractor.Dim(),
			Model:      "campplus_sv_zh-cn",
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
