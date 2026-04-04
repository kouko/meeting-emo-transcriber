package speaker

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"github.com/kouko/meeting-emo-transcriber/internal/audio"
	"github.com/kouko/meeting-emo-transcriber/internal/types"
)

type unknownSpeaker struct {
	name      string
	embedding []float32
}

type Discovery struct {
	store      *Store
	extractor  *Extractor
	matcher    *Matcher
	threshold  float32
	noDiscover bool
	unknowns   []unknownSpeaker
	nextID     int
}

func NewDiscovery(store *Store, extractor *Extractor, matcher *Matcher, threshold float32, noDiscover bool) *Discovery {
	nextID := scanMaxSpeakerID(store.Root()) + 1
	return &Discovery{
		store:      store,
		extractor:  extractor,
		matcher:    matcher,
		threshold:  threshold,
		noDiscover: noDiscover,
		nextID:     nextID,
	}
}

func scanMaxSpeakerID(dir string) int {
	re := regexp.MustCompile(`^speaker_(\d+)$`)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	maxID := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		matches := re.FindStringSubmatch(e.Name())
		if len(matches) == 2 {
			if id, err := strconv.Atoi(matches[1]); err == nil && id > maxID {
				maxID = id
			}
		}
	}
	return maxID
}

func (d *Discovery) IdentifySpeaker(
	embedding []float32,
	profiles []types.SpeakerProfile,
	segAudio []float32,
	sampleRate int,
	segStart float64,
) (string, float32) {
	// 1. Try known profiles
	if len(profiles) > 0 {
		result := d.matcher.Match(embedding, profiles, d.threshold)
		if result.Name != "" {
			return result.Name, result.Similarity
		}
	}

	// 2. Try session unknowns
	for _, u := range d.unknowns {
		sim := CosineSimilarity(embedding, u.embedding)
		if sim >= d.threshold {
			return u.name, sim
		}
	}

	// 3. Create new unknown
	return d.createUnknown(embedding, segAudio, sampleRate, segStart), 0
}

func (d *Discovery) createUnknown(embedding []float32, segAudio []float32, sampleRate int, segStart float64) string {
	id := d.nextID
	d.nextID++

	var name string
	if d.noDiscover {
		name = fmt.Sprintf("Unknown_%d", id)
	} else {
		name = fmt.Sprintf("speaker_%d", id)
		d.persistUnknown(name, embedding, segAudio, sampleRate, segStart)
	}

	d.unknowns = append(d.unknowns, unknownSpeaker{name: name, embedding: embedding})
	return name
}

func (d *Discovery) persistUnknown(name string, embedding []float32, segAudio []float32, sampleRate int, segStart float64) {
	speakerDir := filepath.Join(d.store.Root(), name)
	os.MkdirAll(speakerDir, 0755)

	if len(segAudio) > 0 && sampleRate > 0 {
		timestamp := fmt.Sprintf("%04d", int(segStart*100))
		wavPath := filepath.Join(speakerDir, fmt.Sprintf("auto_segment_%s.wav", timestamp))
		audio.WriteWAV(wavPath, segAudio, sampleRate)
	}

	now := time.Now().Format(time.RFC3339)
	profile := types.SpeakerProfile{
		Name: name,
		Embeddings: []types.SampleEmbedding{
			{
				File:      fmt.Sprintf("auto_segment_%04d.wav", int(segStart*100)),
				Hash:      "",
				Embedding: embedding,
			},
		},
		Dim:       len(embedding),
		Model:     "campplus_sv_zh-cn",
		CreatedAt: now,
		UpdatedAt: now,
	}
	d.store.SaveProfile(profile)
}

func (d *Discovery) Unknowns() []string {
	names := make([]string, len(d.unknowns))
	for i, u := range d.unknowns {
		names[i] = u.name
	}
	return names
}
