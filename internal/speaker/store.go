package speaker

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

// Store manages the folder-driven speaker voiceprint store.
type Store struct {
	root      string
	audioExts map[string]struct{}
}

// NewStore returns a Store rooted at dir, recognising the given audio extensions.
func NewStore(dir string, audioExts []string) *Store {
	extSet := make(map[string]struct{}, len(audioExts))
	for _, e := range audioExts {
		extSet[e] = struct{}{}
	}
	return &Store{root: dir, audioExts: extSet}
}

// Root returns the root directory path of the speaker store.
func (s *Store) Root() string {
	return s.root
}

// List returns the names of all subdirectories in the root (each is a speaker).
func (s *Store) List() ([]string, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []string{}, nil
		}
		return nil, err
	}

	var names []string
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	return names, nil
}

// LoadProfiles loads and merges all *.profile.json files for every speaker.
// Also loads legacy .profile.json for backward compatibility.
func (s *Store) LoadProfiles() ([]types.SpeakerProfile, error) {
	speakers, err := s.List()
	if err != nil {
		return nil, err
	}

	var profiles []types.SpeakerProfile
	for _, name := range speakers {
		p, err := s.LoadProfile(name)
		if err != nil {
			return nil, err
		}
		if p != nil && len(p.Embeddings) > 0 {
			profiles = append(profiles, *p)
		}
	}
	return profiles, nil
}

// LoadProfile loads and merges all *.profile.json files in a speaker's directory.
// Returns nil, nil if no profiles exist.
func (s *Store) LoadProfile(name string) (*types.SpeakerProfile, error) {
	speakerDir := filepath.Join(s.root, name)
	entries, err := os.ReadDir(speakerDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	merged := &types.SpeakerProfile{
		Name: name,
	}
	found := false

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		fname := e.Name()
		if !strings.HasSuffix(fname, ".profile.json") {
			continue
		}

		data, err := os.ReadFile(filepath.Join(speakerDir, fname))
		if err != nil {
			return nil, err
		}
		var p types.SpeakerProfile
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}

		// Merge embeddings
		merged.Embeddings = append(merged.Embeddings, p.Embeddings...)

		// Merge known audio hashes
		merged.KnownAudioHashes = append(merged.KnownAudioHashes, p.KnownAudioHashes...)

		// Track earliest created_at and latest updated_at
		if merged.CreatedAt == "" || (p.CreatedAt != "" && p.CreatedAt < merged.CreatedAt) {
			merged.CreatedAt = p.CreatedAt
		}
		if p.UpdatedAt > merged.UpdatedAt {
			merged.UpdatedAt = p.UpdatedAt
		}

		found = true
	}

	if !found {
		return nil, nil
	}
	return merged, nil
}

// SaveProfile writes a profile to a named file in the speaker's directory.
// filename should be like "YYYYMMDD-UUID.profile.json".
func (s *Store) SaveProfile(name, filename string, profile types.SpeakerProfile) error {
	speakerDir := filepath.Join(s.root, name)
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(speakerDir, filename), data, 0644)
}

// ListAudioFiles returns the paths of audio files inside a speaker's directory.
func (s *Store) ListAudioFiles(speaker string) ([]string, error) {
	speakerDir := filepath.Join(s.root, speaker)
	entries, err := os.ReadDir(speakerDir)
	if err != nil {
		return nil, err
	}

	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		ext := filepath.Ext(name)
		if _, ok := s.audioExts[ext]; ok {
			files = append(files, filepath.Join(speakerDir, name))
		}
	}
	return files, nil
}

// FindNewAudioFiles returns audio files whose hash is NOT in any profile's known_audio_hashes.
// These are files manually added by the user.
func (s *Store) FindNewAudioFiles(speaker string) ([]string, error) {
	profile, err := s.LoadProfile(speaker)
	if err != nil {
		return nil, err
	}

	// Build hash set from all profiles
	knownHashes := make(map[string]bool)
	if profile != nil {
		for _, h := range profile.KnownAudioHashes {
			knownHashes[h] = true
		}
	}

	audioFiles, err := s.ListAudioFiles(speaker)
	if err != nil {
		return nil, err
	}

	var newFiles []string
	for _, path := range audioFiles {
		hash, err := FileHash(path)
		if err != nil {
			return nil, err
		}
		if !knownHashes[hash] {
			newFiles = append(newFiles, path)
		}
	}
	return newFiles, nil
}

// FileHash computes the SHA-256 hash of the file at path and returns it as "sha256:<hex>".
func FileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}
