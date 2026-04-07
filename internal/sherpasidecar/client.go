package sherpasidecar

import (
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

// Client is the main-process handle to a running metr-sherpa sidecar. It
// owns the child process, its stdio pipes, and the gob encoder/decoder.
//
// All exported methods are safe to call from multiple goroutines; an
// internal mutex serialises requests because the protocol is strictly
// synchronous (one request, then read the reply) and the sidecar handles
// messages serially anyway. Concurrent calls will queue, not race.
//
// A Client is single-use: once Close has been called, or once the sidecar
// has died, no further calls will succeed. The main process is expected to
// spawn one Client per transcribe run.
type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	enc    *gob.Encoder
	dec    *gob.Decoder

	mu     sync.Mutex // serialises request/response round trips
	nextID uint64     // atomic counter for envelope IDs

	closeOnce sync.Once
	closed    atomic.Bool

	// dead is set when we observe a non-recoverable stream error (io.EOF,
	// pipe error, etc.). Once set, all subsequent calls fail-fast instead
	// of deadlocking on a pipe write to a dead child.
	dead atomic.Bool
}

// Spawn starts the metr-sherpa binary at binaryPath as a child process.
// The caller is responsible for calling Close when done.
//
// Spawn does not load any models. The caller must call LoadPunctuator
// and/or LoadClassifier afterwards, matching the features it intends to
// use.
func Spawn(binaryPath string) (*Client, error) {
	if _, err := os.Stat(binaryPath); err != nil {
		return nil, fmt.Errorf("metr-sherpa binary not found at %s: %w", binaryPath, err)
	}

	cmd := exec.Command(binaryPath)
	// Forward the sidecar's stderr to our own stderr so its protocol-level
	// errors show up in the user-visible log without extra plumbing.
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("sherpasidecar: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("sherpasidecar: stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return nil, fmt.Errorf("sherpasidecar: start %s: %w", binaryPath, err)
	}

	return &Client{
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
		enc:    gob.NewEncoder(stdin),
		dec:    gob.NewDecoder(stdout),
	}, nil
}

// errClientDead is returned when the sidecar's stdio stream has gone away.
// Callers should treat this as "the sidecar is gone; degrade gracefully".
var errClientDead = errors.New("sherpasidecar: client is dead")

// roundTrip sends one request envelope and reads the matching reply. It
// holds the client mutex for the entire exchange so concurrent callers
// don't interleave bytes on the pipe.
func (c *Client) roundTrip(req *Envelope) (*Envelope, error) {
	if c.dead.Load() {
		return nil, errClientDead
	}
	if c.closed.Load() {
		return nil, errors.New("sherpasidecar: client is closed")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Re-check under the lock in case another goroutine just marked the
	// client dead while we were waiting.
	if c.dead.Load() {
		return nil, errClientDead
	}

	req.ID = atomic.AddUint64(&c.nextID, 1)

	if err := c.enc.Encode(req); err != nil {
		c.markDead()
		return nil, fmt.Errorf("sherpasidecar: encode: %w", err)
	}

	var reply Envelope
	if err := c.dec.Decode(&reply); err != nil {
		c.markDead()
		return nil, fmt.Errorf("sherpasidecar: decode: %w", err)
	}
	if reply.ID != req.ID {
		// The protocol is strictly synchronous so IDs should always match.
		// A mismatch means the stream is out of sync and there's no safe
		// way to recover.
		c.markDead()
		return nil, fmt.Errorf("sherpasidecar: reply id mismatch (got %d, want %d)", reply.ID, req.ID)
	}
	return &reply, nil
}

// markDead records that the stream is unusable. Called from roundTrip on
// any encode/decode failure. Subsequent calls fail fast.
func (c *Client) markDead() {
	c.dead.Store(true)
}

// LoadPunctuator asks the sidecar to load the CT-Transformer punctuation
// model from modelDir. Must be called at most once per Client.
func (c *Client) LoadPunctuator(modelDir string, threads int) error {
	reply, err := c.roundTrip(&Envelope{
		Kind:      MsgLoadPunct,
		LoadPunct: &LoadPunctReq{ModelDir: modelDir, Threads: threads},
	})
	if err != nil {
		return err
	}
	if reply.Kind != MsgLoadPunctAck || reply.LoadPunctAck == nil {
		return fmt.Errorf("sherpasidecar: unexpected reply kind %d for LoadPunct", reply.Kind)
	}
	if reply.LoadPunctAck.Error != "" {
		return errors.New(reply.LoadPunctAck.Error)
	}
	return nil
}

// Punctuate asks the sidecar to punctuate text. language is currently
// informational but is passed through so future model revisions can use
// it. The main-side wrapper still handles fullwidth-to-ASCII conversion.
func (c *Client) Punctuate(text, language string) (string, error) {
	reply, err := c.roundTrip(&Envelope{
		Kind:  MsgPunctuate,
		Punct: &PunctReq{Text: text, Language: language},
	})
	if err != nil {
		return "", err
	}
	if reply.Kind != MsgPunctuateAck || reply.PunctAck == nil {
		return "", fmt.Errorf("sherpasidecar: unexpected reply kind %d for Punctuate", reply.Kind)
	}
	if reply.PunctAck.Error != "" {
		return "", errors.New(reply.PunctAck.Error)
	}
	return reply.PunctAck.Text, nil
}

// LoadClassifier asks the sidecar to load the SenseVoice emotion model
// from modelDir. Must be called at most once per Client.
func (c *Client) LoadClassifier(modelDir string, threads int) error {
	reply, err := c.roundTrip(&Envelope{
		Kind:      MsgLoadClass,
		LoadClass: &LoadClassReq{ModelDir: modelDir, Threads: threads},
	})
	if err != nil {
		return err
	}
	if reply.Kind != MsgLoadClassAck || reply.LoadClassAck == nil {
		return fmt.Errorf("sherpasidecar: unexpected reply kind %d for LoadClass", reply.Kind)
	}
	if reply.LoadClassAck.Error != "" {
		return errors.New(reply.LoadClassAck.Error)
	}
	return nil
}

// ClassifyResult is the flattened emotion result returned by the sidecar.
// Mapping Raw → user-facing label/display happens in the
// internal/emotion package using types.SenseVoiceEmotionMap.
type ClassifyResult struct {
	Raw        string
	AudioEvent string
	Confidence float32
}

// Classify asks the sidecar to run emotion + audio-event classification on
// a chunk of PCM samples.
func (c *Client) Classify(samples []float32, sampleRate int) (ClassifyResult, error) {
	reply, err := c.roundTrip(&Envelope{
		Kind:  MsgClassify,
		Class: &ClassReq{Samples: samples, SampleRate: sampleRate},
	})
	if err != nil {
		return ClassifyResult{}, err
	}
	if reply.Kind != MsgClassifyAck || reply.ClassAck == nil {
		return ClassifyResult{}, fmt.Errorf("sherpasidecar: unexpected reply kind %d for Classify", reply.Kind)
	}
	if reply.ClassAck.Error != "" {
		return ClassifyResult{}, errors.New(reply.ClassAck.Error)
	}
	return ClassifyResult{
		Raw:        reply.ClassAck.Raw,
		AudioEvent: reply.ClassAck.AudioEvent,
		Confidence: reply.ClassAck.Confidence,
	}, nil
}

// IsDead reports whether the sidecar's stream has encountered a
// non-recoverable error. Callers can use this to stop sending further
// requests and fall back to degraded behaviour for remaining work.
func (c *Client) IsDead() bool {
	return c.dead.Load()
}

// Close sends a shutdown envelope, closes stdin to signal EOF, and waits
// for the child process to exit. If the child does not exit within a
// short grace period, it is killed. Close is idempotent.
func (c *Client) Close() error {
	var closeErr error
	c.closeOnce.Do(func() {
		c.closed.Store(true)

		// Best-effort shutdown envelope. Ignore errors — the stream may
		// already be dead, in which case we just fall through to closing
		// stdin and waiting.
		if !c.dead.Load() {
			c.mu.Lock()
			_ = c.enc.Encode(&Envelope{Kind: MsgShutdown})
			c.mu.Unlock()
		}

		_ = c.stdin.Close()

		done := make(chan error, 1)
		go func() { done <- c.cmd.Wait() }()

		select {
		case err := <-done:
			// exec.Cmd.Wait returns *exec.ExitError even for exit code 0 on
			// some platforms if stdio pipes are closed oddly; treat any
			// clean exit as success.
			if err != nil && !isBenignExitErr(err) {
				closeErr = fmt.Errorf("sherpasidecar: wait: %w", err)
			}
		case <-time.After(2 * time.Second):
			_ = c.cmd.Process.Kill()
			<-done
			closeErr = errors.New("sherpasidecar: child did not exit within grace period, killed")
		}

		_ = c.stdout.Close()
	})
	return closeErr
}

// isBenignExitErr filters out "exit status 0" from exec.Cmd.Wait, which on
// macOS can surface as a non-nil error when stdio pipes are torn down in
// an order the child doesn't like. Anything with a non-zero exit code or
// a signal is reported as a real error.
func isBenignExitErr(err error) bool {
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		return false
	}
	return exitErr.ProcessState != nil && exitErr.ProcessState.ExitCode() == 0
}
