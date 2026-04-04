package types

import "testing"

func TestLookupEmotion_SenseVoice(t *testing.T) {
	tests := []struct {
		raw     string
		label   string
		display string
	}{
		{"HAPPY", "Happy", "happily"},
		{"SAD", "Sad", "sadly"},
		{"ANGRY", "Angry", "angrily"},
		{"NEUTRAL", "Neutral", ""},
		{"unk", "Unknown", ""},
		{"NONEXISTENT", "Unknown", ""},
	}
	for _, tt := range tests {
		info := LookupEmotion(tt.raw, SenseVoiceEmotionMap)
		if info.Label != tt.label {
			t.Errorf("LookupEmotion(%q).Label = %q, want %q", tt.raw, info.Label, tt.label)
		}
		if info.Display != tt.display {
			t.Errorf("LookupEmotion(%q).Display = %q, want %q", tt.raw, info.Display, tt.display)
		}
		if info.Raw != tt.raw {
			t.Errorf("LookupEmotion(%q).Raw = %q, want %q", tt.raw, info.Raw, tt.raw)
		}
	}
}

func TestLookupEmotion_Emotion2vec(t *testing.T) {
	info := LookupEmotion("happy", Emotion2vecEmotionMap)
	if info.Label != "Happy" || info.Display != "happily" {
		t.Errorf("got %+v", info)
	}
	info = LookupEmotion("other", Emotion2vecEmotionMap)
	if info.Label != "Other" || info.Display != "" {
		t.Errorf("got %+v", info)
	}
}
