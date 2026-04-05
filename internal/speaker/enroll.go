package speaker

import (
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// BatchVoiceprintFunc extracts speaker voiceprints from multiple WAV files at once.
type BatchVoiceprintFunc func(wavPaths []string) ([][]float32, error)

// AutoEnroll checks all speakers for manually added audio files (not in known_audio_hashes).
// Only computes embeddings for new files. Returns the number of speakers updated.
func AutoEnroll(store *Store, ffmpegPath string, batchExtractFn BatchVoiceprintFunc, forceAll ...bool) (int, error) {
	force := len(forceAll) > 0 && forceAll[0]

	names, err := store.List()
	if err != nil {
		return 0, err
	}

	updated := 0
	for _, name := range names {
		var newFiles []string
		if force {
			// Force: treat all audio files as new
			newFiles, err = store.ListAudioFiles(name)
			if err != nil {
				return updated, err
			}
		} else {
			// Only find files not in known_audio_hashes
			newFiles, err = store.FindNewAudioFiles(name)
			if err != nil {
				return updated, err
			}
		}

		if len(newFiles) == 0 {
			continue
		}

		fmt.Fprintf(os.Stderr, "  Auto-enrolling %s (%d new samples)...\n", name, len(newFiles))

		tmpDir, err := os.MkdirTemp("", "met-enroll-*")
		if err != nil {
			return updated, fmt.Errorf("create temp dir: %w", err)
		}

		// Convert all new audio files to WAV
		var wavPaths []string
		for i, file := range newFiles {
			tempWav := filepath.Join(tmpDir, fmt.Sprintf("enroll_%d.wav", i))
			if err := audio.ConvertToWAV(ffmpegPath, file, tempWav); err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("convert %s: %w", file, err)
			}
			wavPaths = append(wavPaths, tempWav)
		}

		// Batch extract embeddings
		embResults, err := batchExtractFn(wavPaths)
		if err != nil {
			os.RemoveAll(tmpDir)
			return updated, fmt.Errorf("batch extract embeddings for %s: %w", name, err)
		}
		os.RemoveAll(tmpDir)

		// Build voiceprints and collect hashes
		now := time.Now()
		var voiceprints []types.Voiceprint
		var audioHashes []string
		for i, file := range newFiles {
			if i >= len(embResults) {
				break
			}
			hash, err := FileHash(file)
			if err != nil {
				return updated, fmt.Errorf("hash %s: %w", file, err)
			}
			audioHashes = append(audioHashes, hash)

			dim := len(embResults[i])
			voiceprints = append(voiceprints, types.Voiceprint{
				Source:     filepath.Base(file),
				CreatedAt:  now.Format(time.RFC3339),
				Dim:        dim,
				Model:      types.VoiceprintModel,
				Projection: types.VoiceprintProjection,
				Type:       "extracted",
				Vector:     embResults[i],
			})
		}

		// Save as new profile file
		uuid := shortEnrollUUID()
		datePrefix := now.Format("20060102")
		profileFilename := fmt.Sprintf("%s-%s.profile.json", datePrefix, uuid)

		profile := types.SpeakerProfile{
			CreatedAt:        now.Format(time.RFC3339),
			UpdatedAt:        now.Format(time.RFC3339),
			KnownAudioHashes: audioHashes,
			Voiceprints:      voiceprints,
		}

		if err := store.SaveProfile(name, profileFilename, profile); err != nil {
			return updated, fmt.Errorf("save profile %s: %w", name, err)
		}
		updated++
	}

	return updated, nil
}

func shortEnrollUUID() string {
	b := make([]byte, 4)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}
