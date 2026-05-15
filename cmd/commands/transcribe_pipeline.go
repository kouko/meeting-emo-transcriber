package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/config"
	"github.com/kouko/meeting-emo-transcriber/internal/emotion"
	"github.com/kouko/meeting-emo-transcriber/internal/models"
	"github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// classifyEmotionsAndBuildSegments runs the SenseVoice emotion classifier
// over each ASR segment, attaches the matched speaker name, and packages
// everything into a TranscriptResult ready for the output writers. The
// classifier owns a sherpa-onnx model handle; the function takes care of
// loading it and closing it on return.
//
// On classifier-init failure the function returns immediately. Per-segment
// classification errors are not fatal — they degrade that segment to
// "Neutral" + "Speech" but do not interrupt the whole transcript.
func classifyEmotionsAndBuildSegments(
	asrResults []types.ASRResult,
	speakerNames []string,
	wavSamples []float32,
	wavSampleRate int,
	sherpaClient *sherpasidecar.Client,
	threads int,
	inputPath string,
) (types.TranscriptResult, error) {
	emotionModelDir, err := models.EnsureModel("sensevoice-small-int8")
	if err != nil {
		return types.TranscriptResult{}, fmt.Errorf("ensure emotion model: %w", err)
	}
	classifier, err := emotion.NewClassifier(sherpaClient, emotionModelDir, threads)
	if err != nil {
		return types.TranscriptResult{}, fmt.Errorf("init emotion classifier: %w", err)
	}
	defer classifier.Close()

	segments := make([]types.TranscriptSegment, 0, len(asrResults))
	for i, r := range asrResults {
		segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

		emotionInfo := types.EmotionInfo{Label: "Neutral", Display: ""}
		audioEvent := "Speech"
		var emotionConf float32
		if len(segAudio) > 0 {
			emotionResult, event, classErr := classifier.Classify(segAudio, wavSampleRate)
			if classErr == nil {
				emotionInfo = types.EmotionInfo{
					Raw:     emotionResult.Raw,
					Label:   emotionResult.Label,
					Display: emotionResult.Display,
				}
				audioEvent = event
				emotionConf = emotionResult.Confidence
			}
		}

		segments = append(segments, types.TranscriptSegment{
			Start:      r.Start,
			End:        r.End,
			Speaker:    speakerNames[i],
			Emotion:    emotionInfo,
			AudioEvent: audioEvent,
			Language:   r.Language,
			Text:       r.Text,
			Confidence: types.Confidence{Speaker: 0, Emotion: emotionConf},
		})
	}

	speakerSet := make(map[string]bool)
	identified := 0
	for _, seg := range segments {
		speakerSet[seg.Speaker] = true
		if !strings.HasPrefix(seg.Speaker, "speaker_") && seg.Speaker != "Unknown" {
			identified++
		}
	}

	var duration float64
	if len(asrResults) > 0 {
		duration = asrResults[len(asrResults)-1].End
	}

	return types.TranscriptResult{
		Metadata: types.Metadata{
			File:               filepath.Base(inputPath),
			Duration:           time.Duration(duration * float64(time.Second)).String(),
			SpeakersDetected:   len(speakerSet),
			SpeakersIdentified: identified,
			Date:               time.Now().Format(time.RFC3339),
		},
		Segments: segments,
	}, nil
}

// writeTranscriptOutputs renders transcript in every requested format and
// writes each to its destination path. txt/srt outputs are prefixed with
// a UTF-8 BOM so macOS text editors (TextEdit, Subtitles Editor) detect
// the encoding correctly instead of falling back to MacRoman.
//
// format is the same comma-separated value accepted by --format (see
// config.ParseFormats); "all" expands to txt+json+srt.
func writeTranscriptOutputs(
	transcript types.TranscriptResult,
	format, inputPath, outputPath string,
	punctFunc func(string) string,
) error {
	formats := config.ParseFormats(format)
	for _, fmt_ := range formats {
		outPath := resolveOutputPath(inputPath, outputPath, fmt_)
		content, err := formatTranscript(fmt_, transcript, punctFunc)
		if err != nil {
			return fmt.Errorf("format %s: %w", fmt_, err)
		}
		var fileContent []byte
		if fmt_ == "txt" || fmt_ == "srt" {
			fileContent = append([]byte{0xEF, 0xBB, 0xBF}, []byte(content)...)
		} else {
			fileContent = []byte(content)
		}
		if err := os.WriteFile(outPath, fileContent, 0644); err != nil {
			return fmt.Errorf("write %s: %w", outPath, err)
		}
		fmt.Fprintf(os.Stderr, "Written: %s\n", outPath)
	}
	return nil
}
