package speaker

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"

	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

const profileFilename = ".profile.json"

// Store manages the folder-driven speaker voiceprint store.
type Store struct {
	root       string
	audioExts  map[string]struct{}
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
// Returns an empty slice (not an error) if the root does not exist.
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

// LoadProfiles loads all .profile.json files found in speaker subdirectories.
// Speakers that have no .profile.json are silently skipped.
func (s *Store) LoadProfiles() ([]types.SpeakerProfile, error) {
	speakers, err := s.List()
	if err != nil {
		return nil, err
	}

	var profiles []types.SpeakerProfile
	for _, name := range speakers {
		profilePath := filepath.Join(s.root, name, profileFilename)
		data, err := os.ReadFile(profilePath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		var p types.SpeakerProfile
		if err := json.Unmarshal(data, &p); err != nil {
			return nil, err
		}
		profiles = append(profiles, p)
	}
	return profiles, nil
}

// LoadProfile loads a single speaker's profile.
// Returns nil, nil if profile doesn't exist.
func (s *Store) LoadProfile(name string) (*types.SpeakerProfile, error) {
	profilePath := filepath.Join(s.root, name, profileFilename)
	data, err := os.ReadFile(profilePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var profile types.SpeakerProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// SaveProfile writes (or overwrites) the .profile.json for the given speaker,
// creating the speaker subdirectory if necessary.
func (s *Store) SaveProfile(profile types.SpeakerProfile) error {
	speakerDir := filepath.Join(s.root, profile.Name)
	if err := os.MkdirAll(speakerDir, 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath.Join(speakerDir, profileFilename), data, 0644)
}

// ListAudioFiles returns the paths of audio files inside a speaker's directory.
// Only files whose extension is in the supported set are returned; hidden files
// (e.g. .profile.json) are excluded.
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

// NeedsUpdate reports whether the speaker's profile is missing or out of date
// relative to the audio files currently on disk.
//
// Returns true when:
//   - .profile.json does not exist, or
//   - the set of audio files differs from what is recorded in the profile, or
//   - any audio file's SHA-256 hash differs from the cached value.
func (s *Store) NeedsUpdate(speaker string) (bool, error) {
	profilePath := filepath.Join(s.root, speaker, profileFilename)
	data, err := os.ReadFile(profilePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return true, nil
		}
		return false, err
	}

	var profile types.SpeakerProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return true, nil
	}

	audioFiles, err := s.ListAudioFiles(speaker)
	if err != nil {
		return false, err
	}

	// Build a map of basename -> hash from the cached profile.
	cached := make(map[string]string, len(profile.Embeddings))
	for _, e := range profile.Embeddings {
		cached[e.File] = e.Hash
	}

	// If the count differs, an update is needed.
	if len(audioFiles) != len(cached) {
		return true, nil
	}

	// Verify each audio file against the cache.
	for _, path := range audioFiles {
		base := filepath.Base(path)
		cachedHash, ok := cached[base]
		if !ok {
			return true, nil
		}
		current, err := FileHash(path)
		if err != nil {
			return false, err
		}
		if current != cachedHash {
			return true, nil
		}
	}

	return false, nil
}

// FileHash computes the SHA-256 hash of the file at path and returns it as
// "sha256:<hex>".
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
