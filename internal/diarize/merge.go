package diarize

import (
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
// Returns a slice of speaker ID strings parallel to asrResults.
// Returns "" for segments with no diarization overlap.
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
	// Find unique cluster IDs
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

		// Try matching via centroid embedding from diarization
		if len(profiles) > 0 {
			if centroid, ok := diarResult.SpeakerEmbeddings[clusterID]; ok && len(centroid) > 0 {
				centroidF32 := float64sToFloat32s(centroid)
				name = matchAgainstProfiles(centroidF32, profiles, threshold)
			}
		}

		if name == "" {
			// Unmatched → create speaker_N
			name = fmt.Sprintf("speaker_%d", nextUnknownID)
			nextUnknownID++

			// Persist: save representative audio + embedding
			persistUnknownSpeaker(store, name, clusterID, diarResult, wavSamples, sampleRate)
		}

		clusterNames[clusterID] = name
	}

	// Map IDs to names
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

// matchAgainstProfiles compares an embedding against all enrolled profiles.
// Returns the best matching speaker name, or "" if no match above threshold.
func matchAgainstProfiles(embedding []float32, profiles []types.SpeakerProfile, threshold float32) string {
	var bestName string
	var bestSim float32 = -1

	for _, profile := range profiles {
		for _, sample := range profile.Embeddings {
			if len(sample.Embedding) != len(embedding) {
				continue // dimension mismatch (old profile)
			}
			sim := speaker.CosineSimilarity(embedding, sample.Embedding)
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

	// Find longest segment for this speaker and save audio
	var bestSeg Segment
	var bestLen float64
	for _, seg := range diarResult.Segments {
		if seg.Speaker == clusterID {
			segLen := seg.End - seg.Start
			if segLen > bestLen {
				bestLen = segLen
				bestSeg = seg
			}
		}
	}

	segAudio := audio.ExtractSegment(wavSamples, sampleRate, bestSeg.Start, bestSeg.End)
	wavPath := filepath.Join(speakerDir, "auto_sample.wav")
	audio.WriteWAV(wavPath, segAudio, sampleRate)

	// Use centroid embedding from diarization result
	var embeddings []types.SampleEmbedding
	if centroid, ok := diarResult.SpeakerEmbeddings[clusterID]; ok && len(centroid) > 0 {
		embeddings = append(embeddings, types.SampleEmbedding{
			File:      "auto_sample.wav",
			Embedding: float64sToFloat32s(centroid),
		})
	}

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
	store.SaveProfile(profile)
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
