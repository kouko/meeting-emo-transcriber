package diarize

import (
	"crypto/rand"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// AssignSpeakers maps each ASR result to a diarization speaker by maximum time overlap.
func AssignSpeakers(asrResults []types.ASRResult, diarSegments []Segment) []string {
	ids := make([]string, len(asrResults))
	for i, asr := range asrResults {
		var bestSpeaker string
		var bestOverlap float64
		for _, seg := range diarSegments {
			overlap := overlapDuration(asr.Start, asr.End, seg.Start, seg.End)
			if overlap > bestOverlap {
				bestOverlap = overlap
				bestSpeaker = seg.Speaker
			}
		}
		ids[i] = bestSpeaker
	}
	return ids
}

func overlapDuration(s1, e1, s2, e2 float64) float64 {
	start := math.Max(s1, s2)
	end := math.Min(e1, e2)
	if end > start {
		return end - start
	}
	return 0
}

// ResolveSpeakerNames maps diarization speaker IDs to enrolled speaker names
// using WeSpeaker centroid embeddings from the diarization result.
func ResolveSpeakerNames(
	speakerIDs []string,
	diarResult *DiarizeResult,
	wavSamples []float32,
	sampleRate int,
	profiles []types.SpeakerProfile,
	threshold float32,
	store *speaker.Store,
	diarizeBinPath string,
) ([]string, error) {
	clusterSet := make(map[string]bool)
	for _, id := range speakerIDs {
		if id != "" {
			clusterSet[id] = true
		}
	}

	clusterNames := make(map[string]string)
	nextUnknownID := scanMaxSpeakerID(store.Root()) + 1

	for clusterID := range clusterSet {
		name := ""

		// Match via centroid voiceprint
		if len(profiles) > 0 {
			if centroid, ok := diarResult.SpeakerVoiceprints[clusterID]; ok && len(centroid) > 0 {
				centroidF32 := float64sToFloat32s(centroid)
				name = matchAgainstProfiles(centroidF32, profiles, threshold)
			}
		}

		if name == "" {
			name = fmt.Sprintf("speaker_%d", nextUnknownID)
			nextUnknownID++
			persistUnknownSpeaker(store, name, clusterID, diarResult, wavSamples, sampleRate)
		}

		clusterNames[clusterID] = name
	}

	result := make([]string, len(speakerIDs))
	for i, id := range speakerIDs {
		if id != "" {
			result[i] = clusterNames[id]
		} else {
			result[i] = "Unknown"
		}
	}
	return result, nil
}

func matchAgainstProfiles(embedding []float32, profiles []types.SpeakerProfile, threshold float32) string {
	var bestName string
	var bestSim float32 = -1

	for _, profile := range profiles {
		for _, sample := range profile.Voiceprints {
			if len(sample.Vector) != len(embedding) {
				continue
			}
			sim := speaker.CosineSimilarity(embedding, sample.Vector)
			if sim > bestSim {
				bestSim = sim
				bestName = profile.Name
			}
		}
	}

	if bestSim < threshold {
		return ""
	}
	return bestName
}

func float64sToFloat32s(in64 []float64) []float32 {
	out := make([]float32, len(in64))
	for i, v := range in64 {
		out[i] = float32(v)
	}
	return out
}

func persistUnknownSpeaker(store *speaker.Store, name, clusterID string, diarResult *DiarizeResult, wavSamples []float32, sampleRate int) {
	speakerDir := filepath.Join(store.Root(), name)
	os.MkdirAll(speakerDir, 0755)

	now := time.Now()
	datePrefix := now.Format("20060102")
	uuid := shortUUID()

	// Save top 3 longest segments as wav files
	type segWithLen struct {
		seg Segment
		len float64
	}
	var segs []segWithLen
	for _, seg := range diarResult.Segments {
		if seg.Speaker == clusterID {
			segs = append(segs, segWithLen{seg, seg.End - seg.Start})
		}
	}
	// Sort by length descending (simple selection for top 3)
	for i := 0; i < len(segs); i++ {
		for j := i + 1; j < len(segs); j++ {
			if segs[j].len > segs[i].len {
				segs[i], segs[j] = segs[j], segs[i]
			}
		}
	}

	var audioHashes []string
	maxSamples := 3
	if len(segs) < maxSamples {
		maxSamples = len(segs)
	}
	for i := 0; i < maxSamples; i++ {
		segAudio := audio.ExtractSegment(wavSamples, sampleRate, segs[i].seg.Start, segs[i].seg.End)
		wavName := fmt.Sprintf("%s-%s-%d.wav", datePrefix, uuid, i+1)
		wavPath := filepath.Join(speakerDir, wavName)
		audio.WriteWAV(wavPath, segAudio, sampleRate)

		hash, _ := speaker.FileHash(wavPath)
		audioHashes = append(audioHashes, hash)
	}

	// Build profile with centroid voiceprint
	var voiceprints []types.Voiceprint
	if centroid, ok := diarResult.SpeakerVoiceprints[clusterID]; ok && len(centroid) > 0 {
		voiceprints = append(voiceprints, types.Voiceprint{
			Source:     "auto-discover",
			CreatedAt:  now.Format(time.RFC3339),
			Dim:        len(centroid),
			Model:      "wespeaker_v2",
			Projection: "plda_pyannote_community_1",
			Type:       "centroid",
			Vector:     float64sToFloat32s(centroid),
		})
	}

	profile := types.SpeakerProfile{
		CreatedAt:        now.Format(time.RFC3339),
		UpdatedAt:        now.Format(time.RFC3339),
		KnownAudioHashes: audioHashes,
		Voiceprints:      voiceprints,
	}

	profileFilename := fmt.Sprintf("%s-%s.profile.json", datePrefix, uuid)
	store.SaveProfile(name, profileFilename, profile)
}

func shortUUID() string {
	b := make([]byte, 4)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

func scanMaxSpeakerID(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	maxID := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		var id int
		if _, err := fmt.Sscanf(e.Name(), "speaker_%d", &id); err == nil && id > maxID {
			maxID = id
		}
	}
	return maxID
}
