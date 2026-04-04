package speaker

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

func supportedExtensions() []string {
	return []string{".wav", ".mp3", ".m4a", ".flac", ".ogg"}
}

func TestStore_ListEmpty(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 0 {
		t.Errorf("expected empty list, got %v", names)
	}
}

func TestStore_ListWithSpeakers(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "Alice"), 0755)
	os.MkdirAll(filepath.Join(dir, "Bob"), 0755)
	os.WriteFile(filepath.Join(dir, "config.yaml"), []byte(""), 0644)

	store := NewStore(dir, supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Errorf("expected 2 speakers, got %d: %v", len(names), names)
	}
}

func TestStore_ListNonExistentDir(t *testing.T) {
	store := NewStore("/nonexistent/path", supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 0 {
		t.Errorf("expected empty for nonexistent dir, got %v", names)
	}
}

func TestStore_LoadProfiles_NoProfile(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "Alice"), 0755)

	store := NewStore(dir, supportedExtensions())
	profiles, err := store.LoadProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if len(profiles) != 0 {
		t.Errorf("expected 0 profiles, got %d", len(profiles))
	}
}

func TestStore_LoadProfiles_WithProfile(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)

	profile := types.SpeakerProfile{
		Name: "Alice",
		Dim:  512,
		Embeddings: []types.SampleEmbedding{
			{File: "test.wav", Hash: "sha256:abc", Embedding: []float32{1, 0, 0}},
		},
	}
	data, _ := json.MarshalIndent(profile, "", "  ")
	os.WriteFile(filepath.Join(speakerDir, ".profile.json"), data, 0644)

	store := NewStore(dir, supportedExtensions())
	profiles, err := store.LoadProfiles()
	if err != nil {
		t.Fatal(err)
	}
	if len(profiles) != 1 {
		t.Fatalf("expected 1 profile, got %d", len(profiles))
	}
	if profiles[0].Name != "Alice" {
		t.Errorf("expected Alice, got %q", profiles[0].Name)
	}
}

func TestStore_SaveProfile(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, supportedExtensions())

	profile := types.SpeakerProfile{Name: "Bob", Dim: 512}
	err := store.SaveProfile(profile)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "Bob", ".profile.json"))
	if err != nil {
		t.Fatal(err)
	}
	var loaded types.SpeakerProfile
	json.Unmarshal(data, &loaded)
	if loaded.Name != "Bob" {
		t.Errorf("expected Bob, got %q", loaded.Name)
	}
}

func TestStore_ListAudioFiles(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)
	os.WriteFile(filepath.Join(speakerDir, "sample1.wav"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(speakerDir, "sample2.mp3"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(speakerDir, "notes.txt"), []byte("ignore"), 0644)
	os.WriteFile(filepath.Join(speakerDir, ".profile.json"), []byte("{}"), 0644)

	store := NewStore(dir, supportedExtensions())
	files, err := store.ListAudioFiles("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Errorf("expected 2 audio files, got %d: %v", len(files), files)
	}
}

func TestStore_FileHash_Deterministic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.wav")
	os.WriteFile(path, []byte("fake audio data"), 0644)

	h1, err := FileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := FileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Errorf("hash not deterministic: %q != %q", h1, h2)
	}
	if len(h1) < 10 || h1[:7] != "sha256:" {
		t.Errorf("hash format wrong: %q", h1)
	}
}

func TestStore_NeedsUpdate_NoProfile(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)
	os.WriteFile(filepath.Join(speakerDir, "sample.wav"), []byte("data"), 0644)

	store := NewStore(dir, supportedExtensions())
	needs, err := store.NeedsUpdate("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if !needs {
		t.Error("expected NeedsUpdate=true when no .profile.json")
	}
}

func TestStore_NeedsUpdate_Unchanged(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)

	audioPath := filepath.Join(speakerDir, "sample.wav")
	os.WriteFile(audioPath, []byte("audio data"), 0644)

	hash, _ := FileHash(audioPath)
	profile := types.SpeakerProfile{
		Name: "Alice",
		Embeddings: []types.SampleEmbedding{
			{File: "sample.wav", Hash: hash, Embedding: []float32{1, 0, 0}},
		},
	}
	data, _ := json.MarshalIndent(profile, "", "  ")
	os.WriteFile(filepath.Join(speakerDir, ".profile.json"), data, 0644)

	store := NewStore(dir, supportedExtensions())
	needs, err := store.NeedsUpdate("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if needs {
		t.Error("expected NeedsUpdate=false when unchanged")
	}
}

func TestStoreRoot(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	if store.Root() != dir {
		t.Errorf("Root() = %q, want %q", store.Root(), dir)
	}
}
