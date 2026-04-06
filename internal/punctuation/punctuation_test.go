package punctuation

import "testing"

func TestFullwidthToASCII(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "Chinese comma",
			input: "hello，world",
			want:  "hello, world",
		},
		{
			name:  "Chinese period",
			input: "hello。",
			want:  "hello.",
		},
		{
			name:  "Chinese question mark",
			input: "how are you？",
			want:  "how are you?",
		},
		{
			name:  "Mixed punctuation",
			input: "hello，world。how are you？fine！",
			want:  "hello, world. how are you? fine!",
		},
		{
			name:  "No fullwidth chars",
			input: "hello, world.",
			want:  "hello, world.",
		},
		{
			name:  "Empty string",
			input: "",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fullwidthToASCII(tt.input)
			if got != tt.want {
				t.Errorf("fullwidthToASCII(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestIsEnglish(t *testing.T) {
	tests := []struct {
		lang string
		want bool
	}{
		{"en", true},
		{"zh", false},
		{"zh-TW", false},
		{"ja", false},
		{"auto", false},
	}

	for _, tt := range tests {
		t.Run(tt.lang, func(t *testing.T) {
			if got := isEnglish(tt.lang); got != tt.want {
				t.Errorf("isEnglish(%q) = %v, want %v", tt.lang, got, tt.want)
			}
		})
	}
}
