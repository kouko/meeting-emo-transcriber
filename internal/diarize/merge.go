package diarize

import (
	"crypto/rand"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

// matchDetail holds per-profile similarity info for debug logging.
type matchDetail struct {
	ProfileName    string
	BestSim        float32
	NumVoiceprints int
}

// matchResultInfo holds the full matching result with details.
type matchResultInfo struct {
	Matched bool
	Name    string
	BestSim float32
	Details []matchDetail
}

// ResolveSpeakerNames maps diarization speaker IDs to enrolled speaker names
// using WeSpeaker centroid voiceprints from the diarization result.
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
	// Log enrolled profiles
	if len(profiles) > 0 {
		names := make([]string, len(profiles))
		for i, p := range profiles {
			names[i] = p.Name
		}
		fmt.Fprintf(os.Stderr, "  Loaded %d enrolled profiles: %s\n", len(profiles), strings.Join(names, ", "))
	} else {
		fmt.Fprintf(os.Stderr, "  No enrolled profiles found\n")
	}

	// Find unique clusters and compute durations
	clusterSet := make(map[string]bool)
	clusterDuration := make(map[string]float64)
	for _, id := range speakerIDs {
		if id != "" {
			clusterSet[id] = true
		}
	}
	for _, seg := range diarResult.Segments {
		clusterDuration[seg.Speaker] += seg.End - seg.Start
	}

	// Sort cluster IDs for consistent output
	var clusterIDs []string
	for id := range clusterSet {
		clusterIDs = append(clusterIDs, id)
	}
	sort.Strings(clusterIDs)

	fmt.Fprintf(os.Stderr, "  Diarization found %d clusters: %s\n", len(clusterIDs), strings.Join(clusterIDs, ", "))

	clusterNames := make(map[string]string)
	nextUnknownID := scanMaxSpeakerID(store.Root()) + 1
	matchedCount := 0
	newCount := 0

	for _, clusterID := range clusterIDs {
		dur := clusterDuration[clusterID]
		fmt.Fprintf(os.Stderr, "\n  Cluster %s (%.1f min):\n", clusterID, dur/60)

		var result matchResultInfo

		if len(profiles) > 0 {
			if centroid, ok := diarResult.SpeakerVoiceprints[clusterID]; ok && len(centroid) > 0 {
				centroidF32 := float64sToFloat32s(centroid)
				result = matchAgainstProfilesDetailed(centroidF32, profiles, threshold)

				// Print per-profile details
				for _, d := range result.Details {
					fmt.Fprintf(os.Stderr, "    vs %-12s best=%.2f (%d voiceprints)\n", d.ProfileName+":", d.BestSim, d.NumVoiceprints)
				}
			} else {
				fmt.Fprintf(os.Stderr, "    (no voiceprint available for this cluster)\n")
			}
		}

		if result.Matched {
			clusterNames[clusterID] = result.Name
			matchedCount++
			fmt.Fprintf(os.Stderr, "    → matched: %s (sim=%.2f, threshold=%.2f)\n", result.Name, result.BestSim, threshold)
		} else {
			name := fmt.Sprintf("speaker_%d", nextUnknownID)
			nextUnknownID++
			clusterNames[clusterID] = name
			newCount++
			persistUnknownSpeaker(store, name, clusterID, diarResult, wavSamples, sampleRate)
			if result.BestSim > 0 {
				fmt.Fprintf(os.Stderr, "    → no match (best=%.2f < threshold=%.2f), created %s\n", result.BestSim, threshold, name)
			} else {
				fmt.Fprintf(os.Stderr, "    → no enrolled profiles to match, created %s\n", name)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "\n  Summary: %d matched, %d new speakers created\n", matchedCount, newCount)

	// Map IDs to names
	resultNames := make([]string, len(speakerIDs))
	for i, id := range speakerIDs {
		if id != "" {
			resultNames[i] = clusterNames[id]
		} else {
			resultNames[i] = "Unknown"
		}
	}
	return resultNames, nil
}

// matchAgainstProfilesDetailed compares a voiceprint against all enrolled profiles
// and returns detailed per-profile similarity information.
func matchAgainstProfilesDetailed(voiceprint []float32, profiles []types.SpeakerProfile, threshold float32) matchResultInfo {
	var details []matchDetail
	var bestName string
	var bestSim float32 = -1

	for _, profile := range profiles {
		var profileBestSim float32 = -1
		for _, vp := range profile.Voiceprints {
			if len(vp.Vector) != len(voiceprint) {
				continue
			}
			sim := speaker.CosineSimilarity(voiceprint, vp.Vector)
			if sim > profileBestSim {
				profileBestSim = sim
			}
		}

		details = append(details, matchDetail{
			ProfileName:    profile.Name,
			BestSim:        profileBestSim,
			NumVoiceprints: len(profile.Voiceprints),
		})

		if profileBestSim > bestSim {
			bestSim = profileBestSim
			bestName = profile.Name
		}
	}

	matched := bestSim >= threshold
	if !matched {
		bestName = ""
	}

	return matchResultInfo{
		Matched: matched,
		Name:    bestName,
		BestSim: bestSim,
		Details: details,
	}
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
	sort.Slice(segs, func(i, j int) bool { return segs[i].len > segs[j].len })

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
