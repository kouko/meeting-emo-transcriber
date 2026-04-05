package audio

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
)

// audioInfo holds detected audio format metadata.
type audioInfo struct {
	Codec      string
	SampleRate int
	Channels   int
}

// streamRe matches ffmpeg stderr lines like:
// "Stream #0:0: Audio: pcm_s16le (...), 16000 Hz, mono, s16, 256 kb/s"
var streamRe = regexp.MustCompile(`Stream #\d+:\d+.*Audio:\s*(\w+).*?,\s*(\d+)\s*Hz,\s*(\w+)`)

// parseFFmpegInfo extracts codec, sample rate, channels from ffmpeg -i stderr.
func parseFFmpegInfo(stderr string) audioInfo {
	m := streamRe.FindStringSubmatch(stderr)
	if m == nil {
		return audioInfo{}
	}

	codec := m[1]
	rate, _ := strconv.Atoi(m[2])

	var channels int
	switch m[3] {
	case "mono":
		channels = 1
	case "stereo":
		channels = 2
	default:
		// Try parsing as a number (e.g. "5.1" won't parse cleanly, but handle "2" etc.)
		if n, err := strconv.Atoi(m[3]); err == nil {
			channels = n
		}
	}

	return audioInfo{
		Codec:      codec,
		SampleRate: rate,
		Channels:   channels,
	}
}

// isTargetFormat returns true if audio is already 16kHz mono PCM s16le.
func isTargetFormat(info audioInfo) bool {
	return info.Codec == "pcm_s16le" &&
		info.SampleRate == 16000 &&
		info.Channels == 1
}

// DetectFormat runs ffmpeg -i to detect audio format without converting.
func DetectFormat(ffmpegPath, inputPath string) (audioInfo, error) {
	// ffmpeg -i always exits non-zero when no output is specified; capture stderr only.
	cmd := exec.Command(ffmpegPath, "-i", inputPath)
	stderr, _ := cmd.CombinedOutput() // intentionally ignoring exit error
	info := parseFFmpegInfo(string(stderr))
	if info.Codec == "" {
		return audioInfo{}, fmt.Errorf("could not detect audio format for %q", inputPath)
	}
	return info, nil
}

// buildFFmpegArgs constructs conversion arguments:
// -y -i <input> -acodec pcm_s16le -ar 16000 -ac 1 <output>
func buildFFmpegArgs(ffmpegPath, inputPath, outputPath string) []string {
	return []string{
		ffmpegPath,
		"-y",
		"-i", inputPath,
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		outputPath,
	}
}

// ConvertToWAV converts any audio to 16kHz mono PCM WAV.
// If already target format, copies the file instead.
func ConvertToWAV(ffmpegPath, inputPath, outputPath string) error {
	info, err := DetectFormat(ffmpegPath, inputPath)
	if err != nil {
		return fmt.Errorf("DetectFormat: %w", err)
	}

	if isTargetFormat(info) {
		return copyFile(inputPath, outputPath)
	}

	args := buildFFmpegArgs(ffmpegPath, inputPath, outputPath)
	// args[0] is ffmpegPath, rest are the actual arguments
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg conversion failed: %w\noutput: %s", err, string(out))
	}
	return nil
}

// copyFile copies src to dst.
func copyFile(src, dst string) error {
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

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Sync()
}

// ConcatWAVs concatenates multiple WAV files into one using ffmpeg.
// All inputs are resampled to 16kHz mono to ensure compatibility.
func ConcatWAVs(ffmpegPath string, inputs []string, outputPath string) error {
	if len(inputs) == 0 {
		return fmt.Errorf("no input files")
	}
	if len(inputs) == 1 {
		// Single file: just convert to ensure correct format
		return ConvertToWAV(ffmpegPath, inputs[0], outputPath)
	}

	// Build ffmpeg filter_complex concat command
	args := []string{"-y"}
	for _, input := range inputs {
		args = append(args, "-i", input)
	}

	// Build filter: [0:a][1:a][2:a]concat=n=3:v=0:a=1
	filter := ""
	for i := range inputs {
		filter += fmt.Sprintf("[%d:a]", i)
	}
	filter += fmt.Sprintf("concat=n=%d:v=0:a=1", len(inputs))

	args = append(args, "-filter_complex", filter, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", outputPath)

	cmd := exec.Command(ffmpegPath, args...)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg concat failed: %w", err)
	}
	return nil
}
