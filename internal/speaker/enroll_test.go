package speaker

import (
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// fakeAudio replaces ffmpeg-backed audio operations during tests by simply
// copying input bytes into the output path. It records calls for assertions.
type fakeAudio struct {
	mu           sync.Mutex
	convertCalls []string // input paths passed to ConvertToWAV
	convertErr   error
}

func (f *fakeAudio) convert(_ string, in, out string, _ ...audio.ConvertOpts) error {
	f.mu.Lock()
	f.convertCalls = append(f.convertCalls, in)
	f.mu.Unlock()
	if f.convertErr != nil {
		return f.convertErr
	}
	return copyBytes(in, out)
}

func copyBytes(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

// withFakeAudio installs the fake audio ops for the duration of t.
func withFakeAudio(t *testing.T) *fakeAudio {
	t.Helper()
	f := &fakeAudio{}
	prevConvert := convertToWAVFn
	convertToWAVFn = f.convert
	t.Cleanup(func() {
		convertToWAVFn = prevConvert
	})
	return f
}

// recordingExtractor returns a deterministic embedding (or per-call sequence)
// and records the wav paths passed in. If perCall is non-empty it returns
// perCall[N] for the Nth call; otherwise it returns embedding for every call.
type recordingExtractor struct {
	mu        sync.Mutex
	wavPaths  []string
	embedding []float32
	perCall   [][]float32
}

func (r *recordingExtractor) extract(wavPath string) ([]float32, error) {
	r.mu.Lock()
	idx := len(r.wavPaths)
	r.wavPaths = append(r.wavPaths, wavPath)
	r.mu.Unlock()
	if idx < len(r.perCall) {
		return append([]float32(nil), r.perCall[idx]...), nil
	}
	return append([]float32(nil), r.embedding...), nil
}

func writeBytes(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// writeTestWAV writes a sine-wave WAV file of the requested duration so
// AutoEnroll's duration-gating logic can measure it correctly.
func writeTestWAV(t *testing.T, path string, durationSec float64) {
	t.Helper()
	const sampleRate = 16000
	n := int(float64(sampleRate) * durationSec)
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(sampleRate)))
	}
	if err := audio.WriteWAV(path, samples, sampleRate); err != nil {
		t.Fatalf("write wav %s: %v", path, err)
	}
}

func loadProfileFile(t *testing.T, path string) types.SpeakerProfile {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var p types.SpeakerProfile
	if err := json.Unmarshal(data, &p); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return p
}

func TestAutoEnroll_NoSpeakers(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{1, 0, 0}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 0 {
		t.Errorf("enrolled = %d, want 0", n)
	}
	if len(ex.wavPaths) != 0 {
		t.Errorf("extractor called %d times, want 0", len(ex.wavPaths))
	}
}

// AutoEnroll extracts ONE embedding PER audio file (not one from a
// concatenated wav), then L2-normalises and averages them. This is the
// canonical speaker-verification recipe.
func TestAutoEnroll_NewSpeakerMultipleFiles_ExtractsPerFileAndAverages(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Three 6-second wavs → 18s total, above the 15s enrollment minimum.
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 6.0)
	writeTestWAV(t, filepath.Join(speakerDir, "b.wav"), 6.0)
	writeTestWAV(t, filepath.Join(speakerDir, "c.wav"), 6.0)

	store := NewStore(dir, supportedExtensions())
	fake := withFakeAudio(t)
	// Return three different embeddings so we can verify averaging.
	ex := &recordingExtractor{perCall: [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 1 {
		t.Errorf("enrolled = %d, want 1 (Alice)", n)
	}

	// One ConvertToWAV per source file, no ConcatWAVs.
	if len(fake.convertCalls) != 3 {
		t.Errorf("convert calls = %d, want 3 (one per source wav)", len(fake.convertCalls))
	}
	// Extractor called once per file.
	if len(ex.wavPaths) != 3 {
		t.Fatalf("extractor calls = %d, want 3 (one per file)", len(ex.wavPaths))
	}

	profilePath := store.GetSingleProfilePath("Alice")
	if profilePath == "" {
		t.Fatal("no profile written for Alice")
	}
	prof := loadProfileFile(t, profilePath)
	if len(prof.KnownAudioHashes) != 3 {
		t.Errorf("KnownAudioHashes len = %d, want 3", len(prof.KnownAudioHashes))
	}
	if len(prof.Voiceprints) != 1 {
		t.Fatalf("Voiceprints len = %d, want 1 (one merged)", len(prof.Voiceprints))
	}
	if got := prof.Voiceprints[0].Type; got != "merged" {
		t.Errorf("Voiceprint.Type = %q, want \"merged\"", got)
	}

	// Average of three orthogonal unit vectors → (1/√3, 1/√3, 1/√3) after
	// L2 normalisation.
	got := prof.Voiceprints[0].Vector
	want := float32(1.0 / math.Sqrt(3.0))
	for i, v := range got {
		if math.Abs(float64(v-want)) > 1e-4 {
			t.Errorf("merged[%d] = %f, want %f", i, v, want)
		}
	}
	// And the merged vector itself must be unit-norm.
	var norm float32
	for _, v := range got {
		norm += v * v
	}
	if math.Abs(float64(norm)-1.0) > 1e-4 {
		t.Errorf("merged norm² = %f, want 1.0 (L2-normalised)", norm)
	}
}

// Enrollment quality gate: when total audio duration is below the
// EnrollMinDurationSec floor, AutoEnroll warns and skips persisting the
// voiceprint. A short, noisy enrollment would yield an unreliable embedding
// that pollutes downstream matching, so refusing it is safer than warning
// only.
func TestAutoEnroll_TooShortAudio_SkipsEnrollment(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Single 5-second wav — well below the 15-second floor.
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 5.0)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{1, 0, 0}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 0 {
		t.Errorf("enrolled = %d, want 0 (short audio should be skipped)", n)
	}
	// Extractor must not be called because we bail before pass 2.
	if len(ex.wavPaths) != 0 {
		t.Errorf("extractor called %d times, want 0 (gate should skip extraction)", len(ex.wavPaths))
	}
	// No profile.json should be written.
	if path := store.GetSingleProfilePath("Alice"); path != "" {
		t.Errorf("profile written at %s, expected none", path)
	}
}

// Long-enough audio assembled from multiple short files should pass the gate.
func TestAutoEnroll_MultipleShortFilesAddingUp_PassesGate(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Two 8-second files = 16s total, above the 15s floor.
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 8.0)
	writeTestWAV(t, filepath.Join(speakerDir, "b.wav"), 8.0)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{1, 0, 0}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 1 {
		t.Errorf("enrolled = %d, want 1 (16s should pass the gate)", n)
	}
	if len(ex.wavPaths) != 2 {
		t.Errorf("extractor called %d times, want 2 (one per file)", len(ex.wavPaths))
	}
}

func TestAutoEnroll_NoNewFiles_SkipsExtraction(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	wavPath := filepath.Join(speakerDir, "a.wav")
	writeTestWAV(t, wavPath, 20.0)

	hash, err := FileHash(wavPath)
	if err != nil {
		t.Fatal(err)
	}
	existing := types.SpeakerProfile{
		KnownAudioHashes: []string{hash},
		Voiceprints: []types.Voiceprint{
			{Type: "merged", Vector: []float32{1, 0, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	data, _ := json.MarshalIndent(existing, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-cafe.profile.json"), data)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0.6, 0.8, 0}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 0 {
		t.Errorf("enrolled = %d, want 0 (no new files)", n)
	}
	if len(ex.wavPaths) != 0 {
		t.Errorf("extractor calls = %d, want 0 (no new files)", len(ex.wavPaths))
	}
}

func TestAutoEnroll_ForceFlag_ReExtractsEvenWithoutNewFiles(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	wavPath := filepath.Join(speakerDir, "a.wav")
	writeTestWAV(t, wavPath, 20.0)

	hash, _ := FileHash(wavPath)
	existing := types.SpeakerProfile{
		KnownAudioHashes: []string{hash},
		Voiceprints: []types.Voiceprint{
			{Type: "merged", Vector: []float32{1, 0, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	data, _ := json.MarshalIndent(existing, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-cafe.profile.json"), data)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0, 1, 0}}

	n, err := AutoEnroll(store, "fake-ffmpeg", ex.extract, true)
	if err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}
	if n != 1 {
		t.Errorf("enrolled = %d, want 1 (force re-enroll)", n)
	}
	if len(ex.wavPaths) != 1 {
		t.Errorf("extractor calls = %d, want 1 (force)", len(ex.wavPaths))
	}

	// New embedding should be persisted.
	prof := loadProfileFile(t, store.GetSingleProfilePath("Alice"))
	if len(prof.Voiceprints) != 1 {
		t.Fatalf("Voiceprints len = %d, want 1", len(prof.Voiceprints))
	}
	got := prof.Voiceprints[0].Vector
	want := []float32{0, 1, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Vector[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestAverageEmbeddings_OrthogonalUnitVectors(t *testing.T) {
	got := averageEmbeddings([][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	})
	want := float32(1.0 / math.Sqrt(3.0))
	for i, v := range got {
		if math.Abs(float64(v-want)) > 1e-4 {
			t.Errorf("got[%d] = %f, want %f", i, v, want)
		}
	}
}

func TestAverageEmbeddings_RescalesUnnormalisedInputs(t *testing.T) {
	// Two parallel vectors with very different magnitudes should both pull
	// the average toward the same unit direction (not weighted by length).
	got := averageEmbeddings([][]float32{
		{10, 0, 0},
		{0.01, 0, 0},
	})
	if math.Abs(float64(got[0])-1.0) > 1e-4 {
		t.Errorf("got[0] = %f, want 1.0 (each input L2-normalised before averaging)", got[0])
	}
}

func TestAverageEmbeddings_DimMismatchSkipped(t *testing.T) {
	got := averageEmbeddings([][]float32{
		{1, 0, 0},
		{1, 0}, // wrong dim — must be skipped, not panic
	})
	if math.Abs(float64(got[0])-1.0) > 1e-4 {
		t.Errorf("got[0] = %f, want 1.0 (only valid input contributes)", got[0])
	}
}

func TestAverageEmbeddings_AllZeroSkipped(t *testing.T) {
	got := averageEmbeddings([][]float32{
		{0, 0, 0}, // zero norm — skipped
		{1, 0, 0},
	})
	if math.Abs(float64(got[0])-1.0) > 1e-4 {
		t.Errorf("got[0] = %f, want 1.0", got[0])
	}
}

func TestAverageEmbeddings_Empty(t *testing.T) {
	if got := averageEmbeddings(nil); got != nil {
		t.Errorf("got %v, want nil", got)
	}
	if got := averageEmbeddings([][]float32{{}}); got != nil {
		t.Errorf("got %v, want nil for zero-dim input", got)
	}
}

// LOCK-IN: When the existing profile contains a "centroid" voiceprint (e.g. from
// auto-discovery in persistUnknownSpeaker), AutoEnroll keeps it and replaces only
// the "merged" voiceprint. This is intentional and Step 2 must preserve it.
func TestAutoEnroll_PreservesCentroidVoiceprint(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "speaker_1")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 20.0)

	existing := types.SpeakerProfile{
		KnownAudioHashes: nil, // forces AutoEnroll to find the wav as "new"
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Source: "auto-discover", Vector: []float32{0.1, 0.2, 0.3}, Dim: 3, Model: types.VoiceprintModel},
			{Type: "merged", Vector: []float32{0.9, 0.0, 0.0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	data, _ := json.MarshalIndent(existing, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-cafe.profile.json"), data)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0, 0, 1}}

	if _, err := AutoEnroll(store, "fake-ffmpeg", ex.extract); err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}

	prof := loadProfileFile(t, store.GetSingleProfilePath("speaker_1"))
	if len(prof.Voiceprints) != 2 {
		t.Fatalf("Voiceprints len = %d, want 2 (centroid kept + new merged)", len(prof.Voiceprints))
	}

	var hasCentroid, hasMerged bool
	for _, vp := range prof.Voiceprints {
		switch vp.Type {
		case "centroid":
			hasCentroid = true
			if vp.Vector[0] != 0.1 || vp.Vector[1] != 0.2 || vp.Vector[2] != 0.3 {
				t.Errorf("centroid vector mutated: %v", vp.Vector)
			}
		case "merged":
			hasMerged = true
			if vp.Vector[2] != 1 {
				t.Errorf("merged vector not updated: %v", vp.Vector)
			}
		}
	}
	if !hasCentroid {
		t.Error("centroid voiceprint not preserved")
	}
	if !hasMerged {
		t.Error("merged voiceprint not present")
	}
}

// LOCK-IN: Multiple *.profile.json files in one speaker directory get merged
// into a single file with combined voiceprints and audio hashes.
func TestAutoEnroll_MergesMultipleProfileJSONs(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 20.0)

	p1 := types.SpeakerProfile{
		KnownAudioHashes: []string{"sha256:dead"},
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{1, 0, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	p2 := types.SpeakerProfile{
		KnownAudioHashes: []string{"sha256:beef"},
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{0, 1, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	d1, _ := json.MarshalIndent(p1, "", "  ")
	d2, _ := json.MarshalIndent(p2, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-aaaa.profile.json"), d1)
	writeBytes(t, filepath.Join(speakerDir, "20240102-bbbb.profile.json"), d2)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0, 0, 1}}

	if _, err := AutoEnroll(store, "fake-ffmpeg", ex.extract); err != nil {
		t.Fatalf("AutoEnroll: %v", err)
	}

	// After merge: only one *.profile.json should remain.
	entries, _ := os.ReadDir(speakerDir)
	var profiles []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".profile.json") {
			profiles = append(profiles, e.Name())
		}
	}
	if len(profiles) != 1 {
		t.Fatalf("profile.json count = %d, want 1 (merged)", len(profiles))
	}

	prof := loadProfileFile(t, filepath.Join(speakerDir, profiles[0]))
	// Should contain BOTH centroid voiceprints + the new merged.
	gotTypes := []string{}
	for _, vp := range prof.Voiceprints {
		gotTypes = append(gotTypes, vp.Type)
	}
	sort.Strings(gotTypes)
	wantTypes := []string{"centroid", "centroid", "merged"}
	if strings.Join(gotTypes, ",") != strings.Join(wantTypes, ",") {
		t.Errorf("voiceprint types = %v, want %v", gotTypes, wantTypes)
	}

	// CURRENT BEHAVIOR LOCK-IN: KnownAudioHashes is *overwritten* with
	// the current-on-disk audio file hashes — the legacy hashes from the
	// merged profile.json files are *dropped*. (See the writeup in the
	// review: this is the asymmetry between LoadProfile's append and
	// AutoEnroll's overwrite.)
	currentHash, _ := FileHash(filepath.Join(speakerDir, "a.wav"))
	if len(prof.KnownAudioHashes) != 1 || prof.KnownAudioHashes[0] != currentHash {
		t.Errorf("KnownAudioHashes = %v, want exactly [%q]", prof.KnownAudioHashes, currentHash)
	}
}

func TestRebuildSpeaker_DiscardsAllVoiceprintsAndKeepsOnlyMerged(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 20.0)

	// Existing profile has TWO centroids + ONE merged (legacy shape).
	existing := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Type: "centroid", Vector: []float32{1, 0, 0}, Dim: 3, Model: types.VoiceprintModel},
			{Type: "centroid", Vector: []float32{0.9, 0.1, 0}, Dim: 3, Model: types.VoiceprintModel},
			{Type: "merged", Vector: []float32{0.8, 0.2, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	data, _ := json.MarshalIndent(existing, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-old.profile.json"), data)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0, 1, 0}}

	if err := RebuildSpeaker(store, "Alice", "fake-ffmpeg", ex.extract); err != nil {
		t.Fatalf("RebuildSpeaker: %v", err)
	}

	prof := loadProfileFile(t, store.GetSingleProfilePath("Alice"))
	if len(prof.Voiceprints) != 1 {
		t.Fatalf("voiceprints = %d, want exactly 1 (only merged)", len(prof.Voiceprints))
	}
	if prof.Voiceprints[0].Type != "merged" {
		t.Errorf("voiceprint type = %q, want merged", prof.Voiceprints[0].Type)
	}
	// Vector should reflect the new extractor (per-file averaged), not the old [0.8, 0.2, 0].
	got := prof.Voiceprints[0].Vector
	if len(got) != 3 || got[1] < 0.99 {
		t.Errorf("voiceprint vector = %v, want ≈[0,1,0] (new per-file averaged)", got)
	}
}

func TestRebuildSpeaker_CreatesTimestampedBackup(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 20.0)

	original := types.SpeakerProfile{
		Voiceprints: []types.Voiceprint{
			{Type: "merged", Vector: []float32{0.5, 0.5, 0}, Dim: 3, Model: types.VoiceprintModel},
		},
	}
	data, _ := json.MarshalIndent(original, "", "  ")
	writeBytes(t, filepath.Join(speakerDir, "20240101-original.profile.json"), data)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{1, 0, 0}}

	if err := RebuildSpeaker(store, "Alice", "fake-ffmpeg", ex.extract); err != nil {
		t.Fatalf("RebuildSpeaker: %v", err)
	}

	entries, _ := os.ReadDir(speakerDir)
	var backups []string
	for _, e := range entries {
		if strings.Contains(e.Name(), ".bak.") {
			backups = append(backups, e.Name())
		}
	}
	if len(backups) != 1 {
		t.Fatalf("backup files = %v, want exactly 1 (.bak.<timestamp>.json)", backups)
	}
	// Backup should contain the OLD voiceprint (0.5, 0.5, 0), not the new one.
	backed := loadProfileFile(t, filepath.Join(speakerDir, backups[0]))
	if len(backed.Voiceprints) != 1 || backed.Voiceprints[0].Vector[0] != 0.5 {
		t.Errorf("backup voiceprint = %v, want old [0.5, 0.5, 0]", backed.Voiceprints)
	}
}

func TestRebuildSpeaker_NoProfile_StillEnrollsFreshly(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeTestWAV(t, filepath.Join(speakerDir, "a.wav"), 20.0)

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{0, 1, 0}}

	if err := RebuildSpeaker(store, "Alice", "fake-ffmpeg", ex.extract); err != nil {
		t.Fatalf("RebuildSpeaker: %v", err)
	}

	prof := loadProfileFile(t, store.GetSingleProfilePath("Alice"))
	if len(prof.Voiceprints) != 1 || prof.Voiceprints[0].Type != "merged" {
		t.Errorf("voiceprints = %v, want exactly 1 merged", prof.Voiceprints)
	}
}

func TestRebuildSpeaker_NoAudioFiles_Errors(t *testing.T) {
	dir := t.TempDir()
	speakerDir := filepath.Join(dir, "Alice")
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		t.Fatal(err)
	}

	store := NewStore(dir, supportedExtensions())
	withFakeAudio(t)
	ex := &recordingExtractor{embedding: []float32{1, 0, 0}}

	err := RebuildSpeaker(store, "Alice", "fake-ffmpeg", ex.extract)
	if err == nil {
		t.Fatal("RebuildSpeaker: want error for speaker with no audio, got nil")
	}
}
