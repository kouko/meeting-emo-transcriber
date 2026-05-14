package speaker

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// VoiceprintExtractFunc extracts a single voiceprint from a WAV file.
type VoiceprintExtractFunc func(wavPath string) ([]float32, error)

// EnrollMinDurationSec is the minimum total clean-speech duration (across all
// enrolled files for one speaker) required for AutoEnroll to actually compute
// and persist a voiceprint. Industry guidance (Azure: 20s verify / 30s
// identify; NIST SRE: 10-60s) puts the practical floor for reliable
// enrollment around 15 seconds; below that, the embedding becomes too
// session-specific to be useful.
const EnrollMinDurationSec = 15.0

// Package-level seams for tests to override audio operations without ffmpeg.
var (
	convertToWAVFn = audio.ConvertToWAV
)

// averageEmbeddings returns the L2-normalised mean of the input embeddings,
// itself L2-normalised. This is the canonical speaker-verification recipe:
// per-utterance embed → unit-norm → average → unit-norm. It is the
// industry-standard alternative to concatenating audio before extraction —
// concatenation is OOD for x-vector / ECAPA / WeSpeaker models which use
// statistics pooling over short training chunks.
func averageEmbeddings(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return nil
	}
	dim := len(embeddings[0])
	if dim == 0 {
		return nil
	}
	mean := make([]float32, dim)
	for _, emb := range embeddings {
		if len(emb) != dim {
			// Skip mismatched-dim embeddings rather than corrupt the average.
			continue
		}
		var norm float32
		for _, v := range emb {
			norm += v * v
		}
		if norm <= 0 {
			continue
		}
		inv := float32(1.0 / math.Sqrt(float64(norm)))
		for i, v := range emb {
			mean[i] += v * inv
		}
	}
	var norm float32
	for _, v := range mean {
		norm += v * v
	}
	if norm <= 0 {
		return mean
	}
	inv := float32(1.0 / math.Sqrt(float64(norm)))
	for i := range mean {
		mean[i] *= inv
	}
	return mean
}

// AutoEnroll processes all speakers: merges profile jsons, detects new audio
// files, and computes a merged voiceprint by extracting per-file embeddings
// and averaging them on the unit hypersphere. extractFn is called once per
// audio file (in contrast to the older "concat then extract" approach).
//
// Performance note: callers that spawn a subprocess inside extractFn (e.g.
// the metr-diarize sidecar) pay one process startup per audio file. For
// large enrollment sets a batched extractor would amortise model load —
// see diarize.ExtractVoiceprints for the underlying batch capability.
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

		// Step 3: Extract one embedding per audio file and L2-average them.
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

		// Pass 1: convert each source file to wav and measure duration.
		// Skip enrollment early if the total clean speech is too short to
		// produce a reliable voiceprint.
		type convertedWAV struct {
			path     string
			duration float64
		}
		converted := make([]convertedWAV, 0, len(allAudioFiles))
		var totalDuration float64
		for i, file := range allAudioFiles {
			tempWav := filepath.Join(tmpDir, fmt.Sprintf("enroll_%d.wav", i))
			if err := convertToWAVFn(ffmpegPath, file, tempWav); err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("convert %s: %w", file, err)
			}
			samples, rate, err := audio.ReadWAV(tempWav)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  warning: cannot measure duration of %s: %v\n",
					filepath.Base(file), err)
				continue
			}
			if rate == 0 {
				continue
			}
			dur := float64(len(samples)) / float64(rate)
			converted = append(converted, convertedWAV{tempWav, dur})
			totalDuration += dur
		}

		if totalDuration < EnrollMinDurationSec {
			os.RemoveAll(tmpDir)
			fmt.Fprintf(os.Stderr,
				"  warning: %s has only %.1fs of audio (need >= %.1fs), skipping enrollment — add more samples\n",
				name, totalDuration, EnrollMinDurationSec)
			continue
		}

		// Pass 2: extract embeddings from each converted wav.
		embeddings := make([][]float32, 0, len(converted))
		var skipped int
		for _, w := range converted {
			emb, err := extractFn(w.path)
			if err != nil {
				os.RemoveAll(tmpDir)
				return updated, fmt.Errorf("extract voiceprint for %s: %w", w.path, err)
			}
			if len(emb) == 0 {
				skipped++
				continue
			}
			embeddings = append(embeddings, emb)
		}
		os.RemoveAll(tmpDir)

		if len(embeddings) == 0 {
			fmt.Fprintf(os.Stderr, "  warning: no valid embeddings extracted for %s, skipping\n", name)
			continue
		}
		if skipped > 0 {
			fmt.Fprintf(os.Stderr, "  warning: %d/%d files produced empty embeddings for %s\n",
				skipped, len(converted), name)
		}

		emb := averageEmbeddings(embeddings)

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
			Source:     fmt.Sprintf("enroll (%d files)", len(embeddings)),
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
