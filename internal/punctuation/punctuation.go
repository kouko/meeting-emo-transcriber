package punctuation

import (
	"fmt"
	"path/filepath"
	"strings"

	sherpa "github.com/k2-fsa/sherpa-onnx-go-macos"
)

// Punctuator wraps sherpa-onnx OfflinePunctuation with CT-Transformer model.
type Punctuator struct {
	inner *sherpa.OfflinePunctuation
}

// NewPunctuator creates a punctuator from a CT-Transformer model directory.
// modelDir should contain model.int8.onnx.
func NewPunctuator(modelDir string, threads int) (*Punctuator, error) {
	modelPath := filepath.Join(modelDir, "model.int8.onnx")

	config := &sherpa.OfflinePunctuationConfig{}
	config.Model.CtTransformer = modelPath
	config.Model.NumThreads = threads
	config.Model.Debug = 0
	config.Model.Provider = "cpu"

	inner := sherpa.NewOfflinePunctuation(config)
	if inner == nil {
		return nil, fmt.Errorf("failed to create punctuator from %s", modelDir)
	}

	return &Punctuator{inner: inner}, nil
}

// AddPunct adds punctuation to text.
// For English text, converts fullwidth CJK punctuation to ASCII equivalents.
func (p *Punctuator) AddPunct(text, language string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}
	result := p.inner.AddPunct(text)
	if isEnglish(language) {
		result = fullwidthToASCII(result)
	}
	return result
}

// Close releases the underlying sherpa-onnx resources.
func (p *Punctuator) Close() {
	if p.inner != nil {
		sherpa.DeleteOfflinePunc(p.inner)
		p.inner = nil
	}
}

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
