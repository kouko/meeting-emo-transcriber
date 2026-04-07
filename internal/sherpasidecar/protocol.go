// Package sherpasidecar defines the wire protocol shared between the metr
// main process and the metr-sherpa sidecar binary.
//
// The main process (pure Go, no cgo) communicates with the sidecar (which
// links sherpa-onnx-go-macos) over a stdin/stdout pipe using encoding/gob.
// gob is used instead of JSON because emotion classification requests carry
// up to several hundred KB of []float32 PCM samples per call, and gob
// encodes native Go slices roughly an order of magnitude cheaper than
// JSON+base64.
//
// This file must NOT import any package that transitively depends on
// sherpa-onnx-go-macos. It is imported by both the main binary and the
// sidecar, and the main binary must stay cgo-free.
package sherpasidecar

// MsgKind identifies the type of message carried by an Envelope.
type MsgKind uint8

const (
	// MsgLoadPunct asks the sidecar to load the CT-Transformer punctuation
	// model from a local directory. One-shot: calling this twice is an error.
	MsgLoadPunct MsgKind = iota + 1
	// MsgLoadPunctAck is the sidecar's reply to MsgLoadPunct.
	MsgLoadPunctAck

	// MsgPunctuate asks the sidecar to add punctuation to a piece of text.
	// Requires a successful MsgLoadPunct first.
	MsgPunctuate
	// MsgPunctuateAck is the sidecar's reply to MsgPunctuate.
	MsgPunctuateAck

	// MsgLoadClass asks the sidecar to load the SenseVoice emotion model.
	// One-shot.
	MsgLoadClass
	// MsgLoadClassAck is the sidecar's reply to MsgLoadClass.
	MsgLoadClassAck

	// MsgClassify asks the sidecar to classify emotion + audio event for a
	// chunk of PCM samples. Requires a successful MsgLoadClass first.
	MsgClassify
	// MsgClassifyAck is the sidecar's reply to MsgClassify.
	MsgClassifyAck

	// MsgShutdown asks the sidecar to exit cleanly. The sidecar writes no
	// reply; the main process should close stdin and wait for the child to
	// exit.
	MsgShutdown
)

// Envelope is the single wire type exchanged over the pipe. Using one top-
// level type keeps gob type registration trivial: both sides just
// encode/decode *Envelope values in a loop.
//
// The Kind field selects which payload field is set; all others are nil.
// gob encodes nil pointer fields very cheaply so the overhead of unused
// fields is negligible.
type Envelope struct {
	Kind MsgKind

	// ID is an opaque request identifier echoed in the matching ack. It
	// lets a future pipelined client match responses to requests. Today's
	// client is strictly synchronous (one request, then read the ack) so
	// any value works, but always populate it so the wire format is
	// stable.
	ID uint64

	LoadPunct    *LoadPunctReq
	LoadPunctAck *LoadPunctAck

	Punct    *PunctReq
	PunctAck *PunctAck

	LoadClass    *LoadClassReq
	LoadClassAck *LoadClassAck

	Class    *ClassReq
	ClassAck *ClassAck
}

// LoadPunctReq asks the sidecar to load the CT-Transformer punctuation
// model. ModelDir must contain model.int8.onnx.
type LoadPunctReq struct {
	ModelDir string
	Threads  int
}

// LoadPunctAck reports whether the punctuation model was loaded. Error is
// empty on success.
type LoadPunctAck struct {
	Error string
}

// PunctReq carries a single piece of text to punctuate.
type PunctReq struct {
	Text string
	// Language is forwarded so the sidecar can skip work on empty input
	// without a round trip, and so future models can condition on language.
	// Today only the raw text is passed to sherpa; the main-side wrapper
	// still handles fullwidth→ASCII conversion.
	Language string
}

// PunctAck carries the punctuated result.
type PunctAck struct {
	Text  string
	Error string
}

// LoadClassReq asks the sidecar to load the SenseVoice emotion model.
// ModelDir must contain model.int8.onnx and tokens.txt.
type LoadClassReq struct {
	ModelDir string
	Threads  int
}

// LoadClassAck reports whether the emotion model was loaded.
type LoadClassAck struct {
	Error string
}

// ClassReq carries one chunk of audio for emotion + event classification.
// Samples are float32 PCM in [-1, 1], mono.
type ClassReq struct {
	Samples    []float32
	SampleRate int
}

// ClassAck carries the raw emotion tag and audio event reported by sherpa.
// The main-side wrapper is responsible for mapping Raw through
// types.SenseVoiceEmotionMap to produce the user-facing label and display
// string — keeping that logic in the main binary means the sidecar's wire
// format stays minimal and the mapping table ships with the UI code.
type ClassAck struct {
	Raw        string
	AudioEvent string
	Confidence float32
	Error      string
}
