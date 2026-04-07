// Package punctuation adds punctuation to text using the CT-Transformer
// model. The model is not loaded in-process; this package is a thin
// wrapper over a sherpasidecar.Client that owns the underlying sherpa C++
// object in a sidecar subprocess. Keeping the sherpa dependency out of
// the main metr binary is what makes metr a single-file portable binary.
package punctuation

import (
	"fmt"
	"strings"
)

// sidecarClient is the minimal interface the Punctuator needs. Defined in
// this package (not as an exported interface in sherpasidecar) so tests
// can substitute a fake without pulling in the sidecar spawn machinery.
type sidecarClient interface {
	LoadPunctuator(modelDir string, threads int) error
	Punctuate(text, language string) (string, error)
}

// Punctuator adds punctuation to text via a sherpasidecar client. The
// underlying sherpa model lives in the sidecar process.
type Punctuator struct {
	client sidecarClient
}

// NewPunctuator loads the CT-Transformer punctuation model into the
// sidecar and returns a Punctuator that routes AddPunct calls through it.
// The caller is responsible for the Client's lifecycle (spawn + close);
// NewPunctuator will not close the client on error.
func NewPunctuator(client sidecarClient, modelDir string, threads int) (*Punctuator, error) {
	if client == nil {
		return nil, fmt.Errorf("punctuation: client is nil")
	}
	if err := client.LoadPunctuator(modelDir, threads); err != nil {
		return nil, fmt.Errorf("load punctuation model: %w", err)
	}
	return &Punctuator{client: client}, nil
}

// AddPunct adds punctuation to text.
// For English text, converts fullwidth CJK punctuation to ASCII equivalents.
func (p *Punctuator) AddPunct(text, language string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}
	result, err := p.client.Punctuate(text, language)
	if err != nil {
		// Degrade gracefully: if the sidecar call fails mid-run, return
		// the unpunctuated text rather than failing the whole transcribe.
		// The caller already accepts that punctuation is optional.
		return text
	}
	if isEnglish(language) {
		result = fullwidthToASCII(result)
	}
	return result
}

// Close is a no-op kept for API compatibility. The sidecar client owns the
// sherpa model's lifetime, not the Punctuator.
func (p *Punctuator) Close() {}

// isEnglish returns true if the language code indicates English.
func isEnglish(lang string) bool {
	return lang == "en"
}

// fullwidthToASCII converts CJK fullwidth punctuation to ASCII equivalents.
// The CT-Transformer model outputs fullwidth punctuation even for English text.
func fullwidthToASCII(s string) string {
	r := strings.NewReplacer(
		"，", ", ",
		"。", ". ",
		"？", "? ",
		"！", "! ",
		"；", "; ",
		"：", ": ",
	)
	return strings.TrimSpace(r.Replace(s))
}
