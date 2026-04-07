// Package sherpasrv is the sidecar-side request dispatch loop for the
// sherpasidecar wire protocol. It lives in a subpackage (not alongside the
// protocol and client) because it is the ONLY place in the repository that
// imports sherpa-onnx-go-macos, and the main metr binary must stay
// cgo-free. Splitting the cgo-dependent code into its own package is what
// lets the guardrail test in cmd/nodeps_test.go pass.
//
// The server reads Envelope values from an io.Reader (stdin in production),
// processes them, and writes reply Envelopes to an io.Writer (stdout). All
// errors that can be attributed to a single request are reported via the
// Error field of the matching ack envelope; only catastrophic protocol
// errors (malformed gob stream, pipe closed) cause Serve to return.
package sherpasrv

import (
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
	proto "github.com/kouko/meeting-emo-transcriber/internal/sherpasidecar"
)

// Serve runs the request/response loop until the client closes the input
// stream or sends MsgShutdown. It returns nil on clean shutdown.
//
// Serve is synchronous: one request in, one reply out. The sherpa C++
// objects it manages are not thread-safe, so we don't bother with a worker
// pool.
func Serve(in io.Reader, out io.Writer) error {
	dec := gob.NewDecoder(in)
	enc := gob.NewEncoder(out)

	srv := &server{}
	defer srv.close()

	for {
		var env proto.Envelope
		if err := dec.Decode(&env); err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return fmt.Errorf("sidecar: decode envelope: %w", err)
		}

		reply, shouldExit := srv.dispatch(&env)
		if reply != nil {
			if err := enc.Encode(reply); err != nil {
				return fmt.Errorf("sidecar: encode reply: %w", err)
			}
		}
		if shouldExit {
			return nil
		}
	}
}

// server holds the live sherpa C++ handles for the duration of the sidecar
// process. A single server instance handles all requests serially — the
// underlying sherpa objects are not documented as thread-safe.
type server struct {
	mu    sync.Mutex
	punct *sherpa.OfflinePunctuation
	class *sherpa.OfflineRecognizer
}

// dispatch handles one envelope. It returns the reply envelope (nil for
// MsgShutdown) and a flag indicating the loop should exit.
func (s *server) dispatch(env *proto.Envelope) (*proto.Envelope, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch env.Kind {
	case proto.MsgLoadPunct:
		return s.handleLoadPunct(env), false
	case proto.MsgPunctuate:
		return s.handlePunctuate(env), false
	case proto.MsgLoadClass:
		return s.handleLoadClass(env), false
	case proto.MsgClassify:
		return s.handleClassify(env), false
	case proto.MsgShutdown:
		return nil, true
	default:
		// Unknown kind: echo back with no payload so the client sees a reply
		// instead of a hang. The client will surface this as a decode error.
		return &proto.Envelope{
			Kind: env.Kind,
			ID:   env.ID,
		}, false
	}
}

func (s *server) handleLoadPunct(env *proto.Envelope) *proto.Envelope {
	reply := &proto.Envelope{Kind: proto.MsgLoadPunctAck, ID: env.ID, LoadPunctAck: &proto.LoadPunctAck{}}
	if env.LoadPunct == nil {
		reply.LoadPunctAck.Error = "missing LoadPunct payload"
		return reply
	}
	if s.punct != nil {
		reply.LoadPunctAck.Error = "punctuation model already loaded"
		return reply
	}

	modelPath := filepath.Join(env.LoadPunct.ModelDir, "model.int8.onnx")
	config := &sherpa.OfflinePunctuationConfig{}
	config.Model.CtTransformer = modelPath
	config.Model.NumThreads = env.LoadPunct.Threads
	config.Model.Debug = 0
	config.Model.Provider = "cpu"

	inner := sherpa.NewOfflinePunctuation(config)
	if inner == nil {
		reply.LoadPunctAck.Error = fmt.Sprintf("failed to create punctuator from %s", env.LoadPunct.ModelDir)
		return reply
	}
	s.punct = inner
	return reply
}

func (s *server) handlePunctuate(env *proto.Envelope) *proto.Envelope {
	reply := &proto.Envelope{Kind: proto.MsgPunctuateAck, ID: env.ID, PunctAck: &proto.PunctAck{}}
	if env.Punct == nil {
		reply.PunctAck.Error = "missing Punct payload"
		return reply
	}
	if s.punct == nil {
		reply.PunctAck.Error = "punctuation model not loaded"
		return reply
	}
	if strings.TrimSpace(env.Punct.Text) == "" {
		reply.PunctAck.Text = env.Punct.Text
		return reply
	}
	reply.PunctAck.Text = s.punct.AddPunct(env.Punct.Text)
	return reply
}

func (s *server) handleLoadClass(env *proto.Envelope) *proto.Envelope {
	reply := &proto.Envelope{Kind: proto.MsgLoadClassAck, ID: env.ID, LoadClassAck: &proto.LoadClassAck{}}
	if env.LoadClass == nil {
		reply.LoadClassAck.Error = "missing LoadClass payload"
		return reply
	}
	if s.class != nil {
		reply.LoadClassAck.Error = "emotion model already loaded"
		return reply
	}

	modelPath := filepath.Join(env.LoadClass.ModelDir, "model.int8.onnx")
	tokensPath := filepath.Join(env.LoadClass.ModelDir, "tokens.txt")

	config := &sherpa.OfflineRecognizerConfig{}
	config.ModelConfig.SenseVoice.Model = modelPath
	config.ModelConfig.SenseVoice.Language = ""
	config.ModelConfig.SenseVoice.UseInverseTextNormalization = 0
	config.ModelConfig.Tokens = tokensPath
	config.ModelConfig.NumThreads = env.LoadClass.Threads
	config.ModelConfig.Debug = 0
	config.ModelConfig.Provider = "cpu"

	inner := sherpa.NewOfflineRecognizer(config)
	if inner == nil {
		reply.LoadClassAck.Error = fmt.Sprintf("failed to create emotion classifier from %s", env.LoadClass.ModelDir)
		return reply
	}
	s.class = inner
	return reply
}

func (s *server) handleClassify(env *proto.Envelope) *proto.Envelope {
	reply := &proto.Envelope{Kind: proto.MsgClassifyAck, ID: env.ID, ClassAck: &proto.ClassAck{}}
	if env.Class == nil {
		reply.ClassAck.Error = "missing Class payload"
		return reply
	}
	if s.class == nil {
		reply.ClassAck.Error = "emotion model not loaded"
		return reply
	}

	stream := sherpa.NewOfflineStream(s.class)
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(env.Class.SampleRate, env.Class.Samples)
	s.class.Decode(stream)
	result := stream.GetResult()
	if result == nil {
		reply.ClassAck.Error = "no result from emotion classifier"
		return reply
	}

	// SenseVoice returns tags like "<|HAPPY|>" and "<|Speech|>"; strip the
	// markers. The main-side wrapper maps Raw → label/display via
	// types.SenseVoiceEmotionMap.
	reply.ClassAck.Raw = strings.TrimPrefix(strings.TrimSuffix(result.Emotion, "|>"), "<|")
	event := strings.TrimPrefix(strings.TrimSuffix(result.Event, "|>"), "<|")
	if event == "" {
		event = "Speech"
	}
	reply.ClassAck.AudioEvent = event
	reply.ClassAck.Confidence = 0
	return reply
}

// close releases any loaded sherpa objects. Safe to call multiple times.
func (s *server) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.punct != nil {
		sherpa.DeleteOfflinePunc(s.punct)
		s.punct = nil
	}
	if s.class != nil {
		sherpa.DeleteOfflineRecognizer(s.class)
		s.class = nil
	}
}
