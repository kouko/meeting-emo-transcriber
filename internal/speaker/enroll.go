package speaker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// VoiceprintExtractFunc extracts a single voiceprint from a WAV file.
type VoiceprintExtractFunc func(wavPath string) ([]float32, error)

// AutoEnroll processes all speakers: merges profile jsons, detects new audio files,
// and recomputes merged voiceprint by concatenating all wav files.
func AutoEnroll(store *Store, ffmpegPath string, extractFn VoiceprintExtractFunc, forceAll ...bool) (int, error) {
	force := len(forceAll) > 0 && forceAll[0]

	names, err := store.List()
	if err != nil {
		return 0, err
	}

	updated := 0
	for _, name := range names {
		// Step 1: Merge multiple profile.json files into one
		mergedPath, err := store.MergeProfileFiles(name)
		if err != nil {
			return updated, fmt.Errorf("merge profiles for %s: %w", name, err)
		}

		// Step 2: Check if there are new audio files
		newFiles, err := store.FindNewAudioFiles(name)
		if err != nil {
			return updated, err
		}

		if !force && len(newFiles) == 0 {
			continue
		}

		// Step 3: Concatenate ALL wav files and compute merged voiceprint
		allAudioFiles, err := store.ListAudioFiles(name)
		if err != nil {
			return updated, err
		}
		if len(allAudioFiles) == 0 {
			continue
		}

		fmt.Fprintf(os.Stderr, "  Auto-enrolling %s (%d audio files, %d new)...\n", name, len(allAudioFiles), len(newFiles))

		tmpDir, err := os.MkdirTemp("", "met-enroll-*")
		if err != nil {
			return updated, fmt.Errorf("create temp dir: %w", err)
		}

		// Convert all audio files to WAV first
		var wavPaths []string
		for i, file := range allAudioFiles {
			tempWav := filepath.Join(tmpDir, fmt.Sprintf("enroll_%d.wav", i))
			if err := audio.ConvertToWAV(ffmpegPath, file, tempWav); err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("convert %s: %w", file, err)
			}
			wavPaths = append(wavPaths, tempWav)
		}

		// Concatenate all WAVs into one
		mergedWav := filepath.Join(tmpDir, "merged.wav")
		if err := audio.ConcatWAVs(ffmpegPath, wavPaths, mergedWav); err != nil {
			os.RemoveAll(tmpDir)
			return updated, fmt.Errorf("concat wav for %s: %w", name, err)
		}

		// Extract single voiceprint from concatenated audio
		emb, err := extractFn(mergedWav)
		os.RemoveAll(tmpDir)
		if err != nil {
			return updated, fmt.Errorf("extract voiceprint for %s: %w", name, err)
		}

		// Collect all audio hashes
		var allHashes []string
		for _, file := range allAudioFiles {
			hash, err := FileHash(file)
			if err != nil {
				return updated, fmt.Errorf("hash %s: %w", file, err)
			}
			allHashes = append(allHashes, hash)
		}

		// Step 4: Update the profile json
		now := time.Now()
		mergedVoiceprint := types.Voiceprint{
			Source:     fmt.Sprintf("enroll (%d files)", len(allAudioFiles)),
			CreatedAt:  now.Format(time.RFC3339),
			Dim:        len(emb),
			Model:      types.VoiceprintModel,
			Projection: types.VoiceprintProjection,
			Type:       "merged",
			Vector:     emb,
		}

		if mergedPath != "" {
			// Read existing profile, remove old merged, add new merged
			data, err := os.ReadFile(mergedPath)
			if err != nil {
				return updated, fmt.Errorf("read profile %s: %w", mergedPath, err)
			}
			var profile types.SpeakerProfile
			if err := json.Unmarshal(data, &profile); err != nil {
				return updated, fmt.Errorf("parse profile %s: %w", mergedPath, err)
			}

			// Remove old merged voiceprints, keep centroids
			var kept []types.Voiceprint
			for _, vp := range profile.Voiceprints {
				if vp.Type != "merged" {
					kept = append(kept, vp)
				}
			}
			kept = append(kept, mergedVoiceprint)
			profile.Voiceprints = kept
			profile.KnownAudioHashes = allHashes
			profile.UpdatedAt = now.Format(time.RFC3339)

			outData, err := json.MarshalIndent(profile, "", "  ")
			if err != nil {
				return updated, err
			}
			if err := os.WriteFile(mergedPath, outData, 0644); err != nil {
				return updated, err
			}
		} else {
			// No existing profile, create new one
			profile := types.SpeakerProfile{
				CreatedAt:        now.Format(time.RFC3339),
				UpdatedAt:        now.Format(time.RFC3339),
				KnownAudioHashes: allHashes,
				Voiceprints:      []types.Voiceprint{mergedVoiceprint},
			}
			filename := fmt.Sprintf("%s-%s.profile.json", now.Format("20060102"), shortUUID())
			if err := store.SaveProfile(name, filename, profile); err != nil {
				return updated, err
			}
		}

		updated++
	}

	return updated, nil
}
