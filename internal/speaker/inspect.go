package speaker

import (
	"sort"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// InterScore records the highest cosine similarity between the target
// speaker's voiceprint and any voiceprint in another enrolled profile.
// High inter scores indicate the two enrolled speakers sound alike to
// the model, which raises the risk of confusion at matching time.
type InterScore struct {
	OtherName string
	MaxSim    float32
}

// InspectionReport summarises the per-file and inter-speaker similarity
// distribution for a single enrolled speaker. The "safety margin" is the
// difference between the worst intra-class match and the strongest
// inter-class match — a positive margin means the speaker is well
// separated from everyone else in the enrolled set.
type InspectionReport struct {
	SpeakerName  string
	IntraSims    []float32
	IntraMean    float32
	IntraMin     float32
	IntraMax     float32
	Inter        []InterScore // sorted descending by MaxSim
	SafetyMargin float32      // IntraMin - max(Inter[*].MaxSim)
}

// ComputeInspection runs pure statistics over already-extracted embeddings.
// fileEmbeddings is the per-file embedding of the target speaker's audio
// samples; targetVoiceprint is the centroid (typically the "merged"
// voiceprint stored in the profile); otherProfiles is the other enrolled
// speakers used for inter-class comparison.
//
// If fileEmbeddings is empty or targetVoiceprint is nil, the returned
// report has zero intra stats but inter stats are still computed.
func ComputeInspection(
	speakerName string,
	fileEmbeddings [][]float32,
	targetVoiceprint []float32,
	otherProfiles []types.SpeakerProfile,
) InspectionReport {
	report := InspectionReport{SpeakerName: speakerName}

	// Intra: every sample's embedding vs the centroid.
	if len(targetVoiceprint) > 0 && len(fileEmbeddings) > 0 {
		report.IntraSims = make([]float32, 0, len(fileEmbeddings))
		var sum float32
		report.IntraMin = 2 // > any cosine
		report.IntraMax = -2
		for _, emb := range fileEmbeddings {
			if len(emb) == 0 {
				continue
			}
			sim := CosineSimilarity(emb, targetVoiceprint)
			report.IntraSims = append(report.IntraSims, sim)
			sum += sim
			if sim < report.IntraMin {
				report.IntraMin = sim
			}
			if sim > report.IntraMax {
				report.IntraMax = sim
			}
		}
		if n := len(report.IntraSims); n > 0 {
			report.IntraMean = sum / float32(n)
		} else {
			report.IntraMin = 0
			report.IntraMax = 0
		}
	}

	// Inter: target centroid vs every other profile.
	if len(targetVoiceprint) > 0 {
		for _, other := range otherProfiles {
			if other.Name == speakerName {
				continue
			}
			sim := BestSimilarity(targetVoiceprint, other)
			if sim == -1 {
				continue
			}
			report.Inter = append(report.Inter, InterScore{
				OtherName: other.Name,
				MaxSim:    sim,
			})
		}
		sort.Slice(report.Inter, func(i, j int) bool {
			return report.Inter[i].MaxSim > report.Inter[j].MaxSim
		})
	}

	// Safety margin: how far the worst intra match is above the best
	// impostor. Positive ≈ well-separated; negative ≈ at least one
	// other speaker scores higher than the speaker's own worst sample.
	if len(report.IntraSims) > 0 && len(report.Inter) > 0 {
		report.SafetyMargin = report.IntraMin - report.Inter[0].MaxSim
	}

	return report
}

// MergedVoiceprint returns the first voiceprint of type "merged" in the
// profile, or nil if none exists. The merged voiceprint is what AutoEnroll
// computes from per-file embeddings and is the canonical representation
// of the speaker used by the matcher.
func MergedVoiceprint(profile *types.SpeakerProfile) []float32 {
	if profile == nil {
		return nil
	}
	for _, vp := range profile.Voiceprints {
		if vp.Type == "merged" {
			return vp.Vector
		}
	}
	return nil
}
