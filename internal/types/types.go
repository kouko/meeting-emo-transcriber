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

type SpeakerProfile struct {
	Name       string            `json:"name"`
	Embeddings []SampleEmbedding `json:"embeddings"`
	Dim        int               `json:"dim"`
	Model      string            `json:"model"`
	CreatedAt  string            `json:"created_at"`
	UpdatedAt  string            `json:"updated_at"`
}

type SampleEmbedding struct {
	File      string    `json:"file"`
	Hash      string    `json:"hash"`
	Embedding []float32 `json:"embedding"`
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
