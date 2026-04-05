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

// SpeakerProfile stores speaker embeddings from one or more sources.
// Multiple *.profile.json files in a speaker directory are merged on load.
type SpeakerProfile struct {
	Name             string            `json:"-"`                        // speaker directory name (not stored in JSON)
	CreatedAt        string            `json:"created_at"`               // first created
	UpdatedAt        string            `json:"updated_at"`               // last modified
	KnownAudioHashes []string          `json:"known_audio_hashes"`       // hashes of auto-generated wav files
	Embeddings       []SampleEmbedding `json:"embeddings"`               // one or more embeddings from different sources
}

// SampleEmbedding is a single embedding with its provenance metadata.
type SampleEmbedding struct {
	Source    string    `json:"source"`     // source audio file name (e.g., "20260320 1401 Recording.mp3")
	CreatedAt string   `json:"created_at"` // when this embedding was computed
	Dim       int      `json:"dim"`        // embedding dimension (256 for WeSpeaker v2)
	Model     string   `json:"model"`      // model used (e.g., "wespeaker_v2")
	Type      string   `json:"type"`       // "centroid" (from diarization) or "extracted" (from single wav)
	Embedding []float32 `json:"embedding"` // the embedding vector
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
