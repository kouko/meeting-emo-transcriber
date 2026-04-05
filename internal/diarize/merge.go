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
// Returns a slice of speaker IDs (0-indexed) parallel to asrResults.
// Returns -1 for segments with no diarization overlap.
func AssignSpeakers(asrResults []types.ASRResult, diarSegments []Segment) []int {
	ids := make([]int, len(asrResults))
	for i, asr := range asrResults {
		bestSpeaker := -1
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

// ResolveSpeakerNames maps cluster IDs to human-readable names.
func ResolveSpeakerNames(
	speakerIDs []int,
	diarSegments []Segment,
	wavSamples []float32,
	sampleRate int,
	extractor *speaker.Extractor,
	profiles []types.SpeakerProfile,
	matcher *speaker.Matcher,
	threshold float32,
	store *speaker.Store,
) ([]string, error) {
	// Find unique cluster IDs
	clusterSet := make(map[int]bool)
	for _, id := range speakerIDs {
		if id >= 0 {
			clusterSet[id] = true
		}
	}

	// For each cluster, find the longest segment as representative
	clusterNames := make(map[int]string)
	nextUnknownID := scanMaxSpeakerID(store.Root()) + 1

	for clusterID := range clusterSet {
		var bestSeg Segment
		var bestLen float64
		for _, seg := range diarSegments {
			if seg.Speaker == clusterID {
				segLen := seg.End - seg.Start
				if segLen > bestLen {
					bestLen = segLen
					bestSeg = seg
				}
			}
		}

		segAudio := audio.ExtractSegment(wavSamples, sampleRate, bestSeg.Start, bestSeg.End)

		name := ""
		if len(segAudio) > 0 && extractor != nil && len(profiles) > 0 {
			emb, err := extractor.Extract(segAudio, sampleRate)
			if err == nil {
				result := matcher.Match(emb, profiles, threshold)
				if result.Name != "" {
					name = result.Name
				}
			}
		}

		if name == "" {
			name = fmt.Sprintf("speaker_%d", nextUnknownID)
			nextUnknownID++
			persistUnknownSpeaker(store, name, segAudio, sampleRate, extractor)
		}

		clusterNames[clusterID] = name
	}

	// Map IDs to names
	result := make([]string, len(speakerIDs))
	for i, id := range speakerIDs {
		if id >= 0 {
			result[i] = clusterNames[id]
		} else {
			result[i] = "Unknown"
		}
	}
	return result, nil
}

func persistUnknownSpeaker(store *speaker.Store, name string, segAudio []float32, sampleRate int, extractor *speaker.Extractor) {
	speakerDir := filepath.Join(store.Root(), name)
	os.MkdirAll(speakerDir, 0755)

	wavPath := filepath.Join(speakerDir, "auto_sample.wav")
	audio.WriteWAV(wavPath, segAudio, sampleRate)

	var embeddings []types.SampleEmbedding
	if extractor != nil && len(segAudio) > 0 {
		emb, err := extractor.Extract(segAudio, sampleRate)
		if err == nil {
			embeddings = append(embeddings, types.SampleEmbedding{
				File:      "auto_sample.wav",
				Embedding: emb,
			})
		}
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
		Model:      "eres2net_base",
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
