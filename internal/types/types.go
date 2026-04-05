package types

// AudioSegment represents an audio clip extracted from the original recording
type AudioSegment struct {
	Start float64
	End   float64
	Audio []float32
}

// TranscriptSegment is a single segment in the final output.
type TranscriptSegment struct {
	Start      float64     `json:"start"`
	End        float64     `json:"end"`
	Speaker    string      `json:"speaker"`
	Emotion    EmotionInfo `json:"emotion"`
	AudioEvent string      `json:"audio_event"`
	Language   string      `json:"language"`
	Text       string      `json:"text"`
	Confidence Confidence  `json:"confidence"`
}

type Confidence struct {
	Speaker float32 `json:"speaker"`
	Emotion float32 `json:"emotion"`
}

type TranscriptResult struct {
	Metadata Metadata            `json:"metadata"`
	Segments []TranscriptSegment `json:"segments"`
}

type Metadata struct {
	File               string `json:"file"`
	Duration           string `json:"duration"`
	SpeakersDetected   int    `json:"speakers_detected"`
	SpeakersIdentified int    `json:"speakers_identified"`
	Date               string `json:"date"`
}

// SpeakerProfile stores speaker voiceprints from one or more sources.
// Multiple *.profile.json files in a speaker directory are merged on load.
type SpeakerProfile struct {
	Name             string       `json:"-"`                  // speaker directory name (not stored in JSON)
	CreatedAt        string       `json:"created_at"`         // first created
	UpdatedAt        string       `json:"updated_at"`         // last modified
	KnownAudioHashes []string    `json:"known_audio_hashes"` // hashes of auto-generated wav files
	Voiceprints      []Voiceprint `json:"voiceprints"`        // one or more voiceprints from different sources
}

// Voiceprint model constants
const (
	VoiceprintModel      = "fluidaudio_embedding_v1"
	VoiceprintProjection = "none"
)

// Voiceprint is a speaker identity vector with its provenance metadata.
type Voiceprint struct {
	Source     string    `json:"source"`     // source audio file name
	CreatedAt  string   `json:"created_at"` // when this voiceprint was computed
	Dim        int      `json:"dim"`        // vector dimension (256 for WeSpeaker)
	Model      string   `json:"model"`      // embedding model
	Projection string   `json:"projection"` // projection method
	Type       string   `json:"type"`       // "centroid" (from diarization) or "extracted" (from single wav)
	Vector     []float32 `json:"vector"`    // the voiceprint vector
}

type MatchResult struct {
	Name       string
	Similarity float32
}

type EnrollResult struct {
	Name    string
	Samples int
	Status  string // "created" | "updated" | "unchanged"
}

type ASRResult struct {
	Start    float64
	End      float64
	Text     string
	Language string
}
