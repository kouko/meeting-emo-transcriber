package speaker

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func TestDiscoveryKnownSpeaker(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	profiles := []types.SpeakerProfile{
		{Name: "Alice", Embeddings: []types.SampleEmbedding{{Embedding: makeEmbedding(512, 1.0)}}},
	}

	name, conf := d.IdentifySpeaker(makeEmbedding(512, 1.0), profiles, nil, 16000, 0.0)
	if name != "Alice" {
		t.Errorf("expected Alice, got %q", name)
	}
	if conf < 0.9 {
		t.Errorf("expected high confidence, got %f", conf)
	}
}

func TestDiscoveryNewUnknown(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	name, _ := d.IdentifySpeaker(makeEmbedding(512, 1.0), nil, nil, 16000, 1.5)
	if name != "speaker_1" {
		t.Errorf("expected speaker_1, got %q", name)
	}
	if len(d.Unknowns()) != 1 {
		t.Errorf("expected 1 unknown, got %d", len(d.Unknowns()))
	}
}

func TestDiscoveryCrossSegmentClustering(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	emb := makeEmbedding(512, 1.0)
	name1, _ := d.IdentifySpeaker(emb, nil, nil, 16000, 0.0)
	name2, _ := d.IdentifySpeaker(emb, nil, nil, 16000, 5.0)

	if name1 != name2 {
		t.Errorf("clustering failed: %q != %q", name1, name2)
	}
	if name1 != "speaker_1" {
		t.Errorf("expected speaker_1, got %q", name1)
	}
}

func TestDiscoveryMultipleUnknowns(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	name1, _ := d.IdentifySpeaker(makeEmbedding(512, 1.0), nil, nil, 16000, 0.0)
	name2, _ := d.IdentifySpeaker(makeEmbedding(512, -1.0), nil, nil, 16000, 5.0)

	if name1 == name2 {
		t.Errorf("different speakers same name: %q", name1)
	}
	if name1 != "speaker_1" || name2 != "speaker_2" {
		t.Errorf("got %q and %q", name1, name2)
	}
}

func TestDiscoveryNoDiscover(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, true)

	name, _ := d.IdentifySpeaker(makeEmbedding(512, 1.0), nil, nil, 16000, 0.0)
	if name != "Unknown_1" {
		t.Errorf("expected Unknown_1, got %q", name)
	}

	speakerDir := filepath.Join(dir, "Unknown_1")
	if _, err := os.Stat(speakerDir); err == nil {
		t.Error("should not create folder in noDiscover mode")
	}
}

func TestDiscoveryPersistence(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	segAudio := make([]float32, 16000)
	for i := range segAudio {
		segAudio[i] = 0.01
	}

	name, _ := d.IdentifySpeaker(makeEmbedding(512, 1.0), nil, segAudio, 16000, 1.5)

	speakerDir := filepath.Join(dir, name)
	if _, err := os.Stat(speakerDir); err != nil {
		t.Fatalf("speaker folder not created: %v", err)
	}

	profilePath := filepath.Join(speakerDir, ".profile.json")
	if _, err := os.Stat(profilePath); err != nil {
		t.Fatalf(".profile.json not created: %v", err)
	}

	entries, _ := os.ReadDir(speakerDir)
	hasWav := false
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".wav" {
			hasWav = true
			break
		}
	}
	if !hasWav {
		t.Error("no audio sample .wav found")
	}
}

func TestDiscoveryNextIDFromExisting(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "speaker_3"), 0755)

	store := NewStore(dir, []string{".wav"})
	matcher := NewMatcher(&MaxSimilarityStrategy{})
	d := NewDiscovery(store, nil, matcher, 0.6, false)

	name, _ := d.IdentifySpeaker(makeEmbedding(512, 1.0), nil, nil, 16000, 0.0)
	if name != "speaker_4" {
		t.Errorf("expected speaker_4, got %q", name)
	}
}

func makeEmbedding(dim int, value float32) []float32 {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = value
	}
	return emb
}
