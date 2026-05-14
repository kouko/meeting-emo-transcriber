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

// _-prefixed directories are reserved for system use (e.g. _metr for
// config + learn-mode review output) and must never be treated as
// enrolled speakers.
func TestStore_List_SkipsUnderscoreDirs(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "Alice"), 0755)
	os.MkdirAll(filepath.Join(dir, "_metr"), 0755)
	os.MkdirAll(filepath.Join(dir, "_review"), 0755)
	os.MkdirAll(filepath.Join(dir, "Bob"), 0755)
	os.MkdirAll(filepath.Join(dir, "speaker_1"), 0755) // auto-discovered, still a speaker

	store := NewStore(dir, supportedExtensions())
	names, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	got := make(map[string]bool)
	for _, n := range names {
		got[n] = true
	}
	for _, want := range []string{"Alice", "Bob", "speaker_1"} {
		if !got[want] {
			t.Errorf("expected %q in list, missing", want)
		}
	}
	for _, banned := range []string{"_metr", "_review"} {
		if got[banned] {
			t.Errorf("expected %q to be filtered, but appeared in list", banned)
		}
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
		Voiceprints: []types.Voiceprint{
			{Source: "test.wav", Vector: []float32{1, 0, 0}},
		},
	}
	data, _ := json.MarshalIndent(profile, "", "  ")
	os.WriteFile(filepath.Join(speakerDir, "test.profile.json"), data, 0644)

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

// When multiple *.profile.json files reference the same audio hash (which
// happens after manual edits or when older versions wrote duplicates),
// LoadProfile must dedup. Otherwise FindNewAudioFiles sees a longer
// known_hashes list than expected and the merge profile grows unbounded
// over successive auto-enrolls.
func TestStore_LoadProfile_DedupsKnownAudioHashes(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)

	common := "sha256:0123456789abcdef"
	p1 := types.SpeakerProfile{
		KnownAudioHashes: []string{common, "sha256:aaa"},
	}
	p2 := types.SpeakerProfile{
		KnownAudioHashes: []string{common, "sha256:bbb"},
	}
	d1, _ := json.MarshalIndent(p1, "", "  ")
	d2, _ := json.MarshalIndent(p2, "", "  ")
	os.WriteFile(filepath.Join(speakerDir, "20240101-aaaa.profile.json"), d1, 0644)
	os.WriteFile(filepath.Join(speakerDir, "20240102-bbbb.profile.json"), d2, 0644)

	store := NewStore(dir, supportedExtensions())
	prof, err := store.LoadProfile("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if prof == nil {
		t.Fatal("profile is nil")
	}
	// Expect 3 unique hashes, not 4.
	if len(prof.KnownAudioHashes) != 3 {
		t.Errorf("KnownAudioHashes len = %d, want 3 (deduped). Got %v",
			len(prof.KnownAudioHashes), prof.KnownAudioHashes)
	}
	seen := map[string]bool{}
	for _, h := range prof.KnownAudioHashes {
		if seen[h] {
			t.Errorf("duplicate hash in merged profile: %q", h)
		}
		seen[h] = true
	}
}

func TestStore_SaveProfile(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, supportedExtensions())

	profile := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Source: "test.wav", Vector: []float32{1, 0, 0}},
		},
	}
	err := store.SaveProfile("Bob", "test.profile.json", profile)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "Bob", "test.profile.json"))
	if err != nil {
		t.Fatal(err)
	}
	var loaded types.SpeakerProfile
	json.Unmarshal(data, &loaded)
	if len(loaded.Voiceprints) != 1 {
		t.Errorf("expected 1 voiceprint, got %d", len(loaded.Voiceprints))
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

func TestStore_FindNewAudioFiles_NoProfile(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)
	os.WriteFile(filepath.Join(speakerDir, "sample.wav"), []byte("data"), 0644)

	store := NewStore(dir, supportedExtensions())
	newFiles, err := store.FindNewAudioFiles("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if len(newFiles) != 1 {
		t.Errorf("expected 1 new audio file when no profile, got %d", len(newFiles))
	}
}

func TestStore_FindNewAudioFiles_AllKnown(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	os.MkdirAll(speakerDir, 0755)

	audioPath := filepath.Join(speakerDir, "sample.wav")
	os.WriteFile(audioPath, []byte("audio data"), 0644)

	hash, _ := FileHash(audioPath)
	profile := types.SpeakerProfile{
		KnownAudioHashes: []string{hash},
		Voiceprints: []types.Voiceprint{
			{Source: "sample.wav", Vector: []float32{1, 0, 0}},
		},
	}
	data, _ := json.MarshalIndent(profile, "", "  ")
	os.WriteFile(filepath.Join(speakerDir, "test.profile.json"), data, 0644)

	store := NewStore(dir, supportedExtensions())
	newFiles, err := store.FindNewAudioFiles("Alice")
	if err != nil {
		t.Fatal(err)
	}
	if len(newFiles) != 0 {
		t.Errorf("expected 0 new audio files when all known, got %d: %v", len(newFiles), newFiles)
	}
}

func TestStoreRoot(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, []string{".wav"})
	if store.Root() != dir {
		t.Errorf("Root() = %q, want %q", store.Root(), dir)
	}
}
