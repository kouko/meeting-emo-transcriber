package diarize

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/speaker"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// BatchVoiceprintExtractFunc extracts embeddings for a batch of WAV files
// in a single model load. Production-grade extractors (e.g. metr-diarize
// invoked via the sidecar) amortise model startup across all inputs,
// which matters for per-segment re-verification where the per-segment
// count can be in the hundreds.
type BatchVoiceprintExtractFunc func(wavPaths []string) ([][]float32, error)

// RefineSpeakerNamesPerSegment re-verifies each ASR segment whose cluster
// was mapped to an enrolled name. For every such segment it extracts a
// per-segment embedding and compares it against the matched profile; if
// cosine similarity falls below threshold, that single segment is demoted
// to "Unknown".
//
// This catches scenarios where diarization merged a stray utterance from
// another speaker into Alice's cluster — without re-verification, that
// utterance gets misattributed to Alice in the transcript. With
// re-verification, the cluster as a whole is still labelled Alice but
// the offending segment becomes Unknown, which is recoverable.
//
// Pass-through cases (no work performed):
//   - cluster-level label is not an enrolled name (e.g. "speaker_3",
//     "Unknown", or empty)
//   - segment shorter than minDurationSec
//   - extractor returned an empty embedding for that segment
//
// threshold should typically be looser than the cluster-level threshold
// (e.g. 0.50 if cluster threshold is 0.65) — a strict-cluster /
// loose-segment combination accepts confident cluster assignments while
// rejecting only the segments that clearly do not belong.
//
// On error from batchExtract the original speakerNames are returned
// unchanged together with the error, so callers can decide whether to
// fall back to the unverified labels.
func RefineSpeakerNamesPerSegment(
	speakerNames []string,
	asrResults []types.ASRResult,
	profiles []types.SpeakerProfile,
	wavSamples []float32,
	sampleRate int,
	batchExtract BatchVoiceprintExtractFunc,
	threshold float32,
	minDurationSec float64,
	tmpDir string,
) ([]string, error) {
	if len(speakerNames) != len(asrResults) {
		return nil, fmt.Errorf("len(speakerNames)=%d != len(asrResults)=%d",
			len(speakerNames), len(asrResults))
	}

	refined := make([]string, len(speakerNames))
	copy(refined, speakerNames)

	if batchExtract == nil || len(profiles) == 0 {
		return refined, nil
	}

	// Build profile lookup by name; only enrolled names participate.
	profMap := make(map[string]types.SpeakerProfile)
	for _, p := range profiles {
		profMap[p.Name] = p
	}

	type pending struct {
		idx     int
		wavPath string
	}
	var todo []pending

	for i, name := range speakerNames {
		if _, isEnrolled := profMap[name]; !isEnrolled {
			continue
		}
		seg := asrResults[i]
		if seg.End-seg.Start < minDurationSec {
			continue
		}
		segAudio := audio.ExtractSegment(wavSamples, sampleRate, seg.Start, seg.End)
		if len(segAudio) == 0 {
			continue
		}
		wavPath := filepath.Join(tmpDir, fmt.Sprintf("verify_%d.wav", i))
		if err := audio.WriteWAV(wavPath, segAudio, sampleRate); err != nil {
			return refined, fmt.Errorf("write segment wav: %w", err)
		}
		todo = append(todo, pending{i, wavPath})
	}

	if len(todo) == 0 {
		return refined, nil
	}

	paths := make([]string, len(todo))
	for i, t := range todo {
		paths[i] = t.wavPath
	}
	embeddings, err := batchExtract(paths)
	if err != nil {
		return refined, fmt.Errorf("batch extract: %w", err)
	}
	if len(embeddings) != len(todo) {
		return refined, fmt.Errorf("batch extract returned %d embeddings for %d inputs",
			len(embeddings), len(todo))
	}

	var demoted int
	for i, t := range todo {
		emb := embeddings[i]
		if len(emb) == 0 {
			continue
		}
		profile := profMap[speakerNames[t.idx]]
		sim := speaker.BestSimilarity(emb, profile)
		if sim < threshold {
			refined[t.idx] = "Unknown"
			demoted++
		}
	}

	if demoted > 0 {
		fmt.Fprintf(os.Stderr,
			"  [re-verify] %d/%d enrolled segments demoted to Unknown (threshold=%.2f)\n",
			demoted, len(todo), threshold)
	}

	return refined, nil
}
