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
// distribution for a single enrolled speaker.
//
// Two safety margins are reported:
//
//   - SafetyMargin: intra-min vs the single "merged" voiceprint in the
//     profile, minus the strongest inter-class score. Highlights when the
//     merged centroid no longer represents the speaker's actual files
//     (e.g. left over from a pre-overhaul concatenation recipe).
//   - BestOfAllSafetyMargin: intra-min vs the BEST-matching voiceprint
//     stored in the target profile (max over all voiceprints), minus the
//     strongest inter-class score. Mirrors the live matcher's
//     BestSimilarity scoring and is the realistic confidence indicator
//     for what the transcribe pipeline will do.
type InspectionReport struct {
	SpeakerName string

	// Per-file similarity vs the merged voiceprint (single stored
	// centroid). May be misleading when the merged voiceprint is stale.
	IntraSims []float32
	IntraMean float32
	IntraMin  float32
	IntraMax  float32

	// Per-file similarity vs the BEST matching voiceprint in the target
	// profile (max over all stored voiceprints). Mirrors live matcher.
	BestOfAllIntraSims []float32
	BestOfAllIntraMean float32
	BestOfAllIntraMin  float32
	BestOfAllIntraMax  float32

	Inter []InterScore // sorted descending by MaxSim (merged voiceprint vs others)

	// BestOfAllInterMax mirrors live matcher: the highest cosine any
	// per-file embedding scores against any other profile, via
	// BestSimilarity. Tightly correlated with the EER the matcher will
	// see on this speaker's audio.
	BestOfAllInterMax float32

	// SafetyMargin = IntraMin − max(Inter[*].MaxSim). Merged-only.
	SafetyMargin float32
	// BestOfAllSafetyMargin = BestOfAllIntraMin − BestOfAllInterMax.
	// Matches live matcher behaviour (per-file emb vs best of any
	// profile's stored voiceprints).
	BestOfAllSafetyMargin float32
}

// ComputeInspection runs pure statistics over already-extracted embeddings.
// fileEmbeddings is the per-file embedding of the target speaker's audio
// samples; targetProfile carries the speaker's stored voiceprints (merged
// + any centroids); otherProfiles is the other enrolled speakers used for
// inter-class comparison.
//
// If fileEmbeddings is empty or targetProfile has no voiceprints, the
// returned report has zero intra stats but inter stats are still
// computed against the merged voiceprint (when present).
func ComputeInspection(
	speakerName string,
	fileEmbeddings [][]float32,
	targetProfile types.SpeakerProfile,
	otherProfiles []types.SpeakerProfile,
) InspectionReport {
	report := InspectionReport{SpeakerName: speakerName}
	merged := MergedVoiceprint(&targetProfile)

	report.IntraSims, report.IntraMean, report.IntraMin, report.IntraMax =
		computeMergedIntra(fileEmbeddings, merged)
	report.BestOfAllIntraSims, report.BestOfAllIntraMean, report.BestOfAllIntraMin, report.BestOfAllIntraMax =
		computeBestOfAllIntra(fileEmbeddings, targetProfile)
	report.Inter = computeInterMergedSorted(speakerName, merged, otherProfiles)
	report.BestOfAllInterMax = computeBestOfAllInterMax(speakerName, fileEmbeddings, otherProfiles)

	if len(report.Inter) > 0 && len(report.IntraSims) > 0 {
		report.SafetyMargin = report.IntraMin - report.Inter[0].MaxSim
	}
	if len(report.BestOfAllIntraSims) > 0 && len(otherProfiles) > 0 {
		report.BestOfAllSafetyMargin = report.BestOfAllIntraMin - report.BestOfAllInterMax
	}
	return report
}

// computeMergedIntra returns per-file cosine vs the merged voiceprint,
// plus aggregate mean/min/max. Skips empty embeddings.
func computeMergedIntra(fileEmbeddings [][]float32, merged []float32) ([]float32, float32, float32, float32) {
	if len(merged) == 0 || len(fileEmbeddings) == 0 {
		return nil, 0, 0, 0
	}
	sims := make([]float32, 0, len(fileEmbeddings))
	var sum, min, max float32 = 0, 2, -2
	for _, emb := range fileEmbeddings {
		if len(emb) == 0 {
			continue
		}
		sim := CosineSimilarity(emb, merged)
		sims = append(sims, sim)
		sum += sim
		if sim < min {
			min = sim
		}
		if sim > max {
			max = sim
		}
	}
	if len(sims) == 0 {
		return sims, 0, 0, 0
	}
	return sims, sum / float32(len(sims)), min, max
}

// computeBestOfAllIntra returns per-file cosine vs the BEST matching
// voiceprint in the target profile (BestSimilarity), plus aggregates.
// Mirrors live matcher scoring.
func computeBestOfAllIntra(fileEmbeddings [][]float32, targetProfile types.SpeakerProfile) ([]float32, float32, float32, float32) {
	if len(targetProfile.Voiceprints) == 0 || len(fileEmbeddings) == 0 {
		return nil, 0, 0, 0
	}
	sims := make([]float32, 0, len(fileEmbeddings))
	var sum, min, max float32 = 0, 2, -2
	for _, emb := range fileEmbeddings {
		if len(emb) == 0 {
			continue
		}
		sim := BestSimilarity(emb, targetProfile)
		if sim == -1 {
			continue
		}
		sims = append(sims, sim)
		sum += sim
		if sim < min {
			min = sim
		}
		if sim > max {
			max = sim
		}
	}
	if len(sims) == 0 {
		return sims, 0, 0, 0
	}
	return sims, sum / float32(len(sims)), min, max
}

// computeInterMergedSorted returns inter-class scores (merged voiceprint
// vs every other profile via BestSimilarity), sorted descending by sim.
func computeInterMergedSorted(speakerName string, merged []float32, otherProfiles []types.SpeakerProfile) []InterScore {
	if len(merged) == 0 {
		return nil
	}
	var out []InterScore
	for _, other := range otherProfiles {
		if other.Name == speakerName {
			continue
		}
		sim := BestSimilarity(merged, other)
		if sim == -1 {
			continue
		}
		out = append(out, InterScore{OtherName: other.Name, MaxSim: sim})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].MaxSim > out[j].MaxSim })
	return out
}

// computeBestOfAllInterMax returns the largest cosine any per-file
// embedding scores against any other profile via BestSimilarity. This
// is the realistic worst case the live matcher will see when an
// utterance from this speaker could be misclassified into another
// profile. Returns 0 when nothing comparable is found.
func computeBestOfAllInterMax(speakerName string, fileEmbeddings [][]float32, otherProfiles []types.SpeakerProfile) float32 {
	var max float32 = -2
	for _, emb := range fileEmbeddings {
		if len(emb) == 0 {
			continue
		}
		for _, other := range otherProfiles {
			if other.Name == speakerName {
				continue
			}
			sim := BestSimilarity(emb, other)
			if sim == -1 {
				continue
			}
			if sim > max {
				max = sim
			}
		}
	}
	if max < -1 {
		return 0
	}
	return max
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
