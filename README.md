# Meeting Emotion Transcriber (metr)

macOS CLI tool (Go) for meeting transcription with speaker identification and emotion recognition.

Combines whisper.cpp (ASR), FluidAudio (diarization), WeSpeaker (speaker embedding), SenseVoice (emotion), and CT-Transformer (punctuation) into a single binary. All processing runs locally — no cloud uploads.

## Features

- **Speech-to-Text** — whisper.cpp with Metal GPU acceleration, supports zh-TW/zh/en/ja
- **Punctuation Restoration** — CT-Transformer auto-adds punctuation for ZH/EN text (applied on TXT output)
- **Speaker Diarization** — FluidAudio `metr-diarize` Swift CLI with CoreML/ANE acceleration
- **Speaker Identification** — WeSpeaker 256-dim voiceprint matching against enrolled profiles
- **Emotion Recognition** — SenseVoice detects Happy/Sad/Angry/Neutral + audio events (Laughter, Applause, etc.)
- **Audio Enhancement** — Optional DeepFilterNet3 noise reduction (`--enhance`)
- **Clipping Protection** — Auto-detects 32-bit Float audio exceeding 0dB and attenuates before conversion
- **Multiple Output Formats** — TXT (merged paragraphs with punctuation), JSON (structured), SRT (subtitles)
- **Single Binary** — whisper-cli + ffmpeg + metr-diarize + metr-denoise embedded via `embed.FS`
- **Content-based Cache** — Skips Whisper on repeated runs (keyed by audio content fingerprint + language)
- **Auto-Enroll** — Automatically re-enrolls speakers when new audio samples are added
- **Portable Mode** — If `./metr-speakers/` exists, uses it instead of `~/metr-speakers/`
- **Auto Speaker Prompt** — Enrolled speaker names are automatically injected into Whisper prompt

## Quick Start

```bash
# Build
git clone https://github.com/kouko/meeting-emo-transcriber.git
cd meeting-emo-transcriber
brew install ffmpeg
make all

# Transcribe (simplest form)
metr meeting.mp3

# With options
metr meeting.mp3 --language zh-TW --format all
metr meeting.mp3 --enhance --normalize
metr meeting.mp3 --learning-mode
```

Models are automatically downloaded on first run (~3.4 GB total).

## Speaker Setup

```
~/metr-speakers/
├── _metr/                        # Resources (bin, models, cache, config)
│   └── config.yaml               # Settings (auto-generated with comments)
├── Alice/
│   ├── office.wav                # Voice sample 1
│   ├── phone.m4a                 # Voice sample 2
│   └── 20260406-a1b2c3d4.profile.json  # Auto-generated voiceprint
├── Bob/
│   └── recording.wav
└── speaker_0/                    # Auto-created unknown speaker
    └── cluster_segment_001.wav
```

### Managing Speakers

After transcription, unknown speakers appear as `speaker_0/`, `speaker_1/`, etc.

**Rename** a speaker:
```bash
mv ~/metr-speakers/speaker_0 ~/metr-speakers/Alice
```

**Merge** two speakers (when the same person was split into multiple clusters):
```bash
mv ~/metr-speakers/speaker_1/*.wav ~/metr-speakers/Alice/
mv ~/metr-speakers/speaker_1/*.profile.json ~/metr-speakers/Alice/
rm -r ~/metr-speakers/speaker_1
```

**Re-run** the same audio file to apply changes — speakers will be re-identified with updated names.

**Learning mode** (`--learning-mode` / `-L`): Creates folders for ALL detected speakers (including already matched ones) so you can review and correct assignments:
```bash
metr meeting.mp3 --learning-mode
```

## Speakers Directory

| Priority | Path | When |
|----------|------|------|
| 1 | `--speakers <path>` | Explicitly specified |
| 2 | `./metr-speakers/` | Exists in current directory (portable mode) |
| 3 | `~/metr-speakers/` | Default (global, shared across all directories) |

Use `metr pack` / `metr unpack` to copy resources between portable and global mode.

## CLI Usage

### `metr <audio file>` — Quick transcribe

```bash
metr meeting.mp3                            # Transcribe with defaults
metr meeting.mp3 --language ja              # Specify language
metr meeting.mp3 --format all               # Output txt + json + srt
metr meeting.mp3 --enhance                  # Enable noise reduction (DeepFilterNet3)
metr meeting.mp3 --normalize                # Force loudnorm normalization
metr meeting.mp3 --prompt "Alice, ACME"     # Custom vocabulary hints for ASR
metr meeting.mp3 --threshold 0.6            # Fewer speakers (lower = more merging)
metr meeting.mp3 --match-threshold 0.7      # Stricter speaker matching
metr meeting.mp3 --learning-mode            # Create folders for all detected speakers
metr meeting.mp3 --no-cache                 # Force re-transcription (skip cache)
```

### Flags

| Flag | Short | Default | Description |
|------|:---:|---------|-------------|
| `--input` | | *(required)* | Audio file path |
| `--output` | | *(auto)* | Output path (auto-appends format extension) |
| `--format` | | `txt` | `txt`, `json`, `srt`, `all`, or comma-separated |
| `--language` | `-l` | `auto` | `auto`, `zh-TW`, `zh`, `en`, `ja` |
| `--threshold` | | `0.80` | Diarization clustering threshold (higher = more speakers) |
| `--match-threshold` | | `0.55` | Speaker matching cosine similarity threshold |
| `--num-speakers` | | `0` | Expected number of speakers (0 = auto-detect) |
| `--learning-mode` | `-L` | `false` | Create folders for all detected speakers for review |
| `--enhance` | | `false` | DeepFilterNet3 noise reduction before processing |
| `--normalize` | | `false` | Force loudnorm normalization (auto-attenuates >0dB regardless) |
| `--no-cache` | | `false` | Skip ASR cache and force re-transcription |
| `--prompt` | | | Custom vocabulary/context hints for ASR (comma-separated) |
| `--speakers` | | `~/metr-speakers` | Speakers directory path |

### Other Commands

```bash
metr init                          # Initialize speakers directory + config template
metr enroll [--force]              # Compute speaker embeddings from audio samples
metr speakers list                 # List registered speakers
metr pack                          # Copy ~/.metr → metr-speakers/_metr/ (portable)
metr unpack                        # Copy metr-speakers/_metr/ → ~/.metr/ (local)
```

## Configuration

`<speakers-dir>/_metr/config.yaml` — auto-generated on first run with detailed comments.
See [example.config.yaml](example.config.yaml) for all options and defaults.

Settings priority: CLI flags > config.yaml > built-in defaults.
Only explicitly-set CLI flags update config.yaml.

## Processing Pipeline

```
metr meeting.mp3
  │
  [1/9] Extract embedded binaries    whisper-cli + ffmpeg + metr-diarize + metr-denoise
  [2/9] Ensure ASR model             language → model (Breeze/Belle/Kotoba/Large-v3)
  [3/9] Convert audio to WAV         ffmpeg → 16kHz mono (auto clipping protection)
  [*]   Enhance audio (optional)     DeepFilterNet3 noise reduction (--enhance)
  [4/9] Speech recognition           whisper-cli → SRT → cache → []ASRResult
  [5/9] Load punctuation model       CT-Transformer ZH+EN (skip for JA)
  [6/9] Speaker diarization          metr-diarize (FluidAudio CoreML/ANE)
  [7/9] Resolve speaker identities   WeSpeaker 256-dim cosine similarity matching
  [8/9] Emotion classification       SenseVoice-Small int8 per segment
  [9/9] Write output files           TXT (with punctuation) / JSON / SRT
```

## Output Examples

### TXT

```
Alice [happily]
Today's quarterly data is very impressive, the growth rate exceeded expectations.

Bob
Let me report the specific numbers, we achieved 120% of the target.

speaker_1 [angrily]
Unit B's problems are still not resolved, we need to address this immediately.
```

### JSON

```json
{
  "metadata": {
    "file": "meeting.wav",
    "duration": "5m30s",
    "speakers_detected": 3,
    "speakers_identified": 2
  },
  "segments": [
    {
      "start": 0.0, "end": 5.2,
      "speaker": "Alice",
      "emotion": {"raw": "HAPPY", "label": "Happy", "display": "happily"},
      "audio_event": "Speech",
      "text": "Today's quarterly data is very impressive",
      "confidence": {"speaker": 0, "emotion": 0.95}
    }
  ]
}
```

### SRT

```
1
00:00:00,000 --> 00:00:05,200
(Alice) [happily] Today's quarterly data is very impressive.
```

## Prerequisites

- macOS ARM64 (Apple Silicon)
- Go 1.25+
- Swift 6.0+ (for building `metr-diarize` and `metr-denoise`)
- Homebrew (`brew install ffmpeg`)

## Build

```bash
make all        # build Swift CLIs + deps + Go binary
make build      # build Go binary only
make swift      # build metr-diarize + metr-denoise
make test       # run all tests
make clean      # remove binary
make clean-all  # remove binary + deps + build cache
make info       # show platform and deps status
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Main CLI | Go 1.25 + cobra + viper |
| ASR | whisper.cpp v1.7.3 (Metal GPU, embedded binary) |
| Punctuation | CT-Transformer ZH+EN int8 via sherpa-onnx (cgo, CPU) |
| Diarization | FluidAudio via `metr-diarize` Swift CLI (CoreML/ANE) |
| Speaker Embedding | WeSpeaker 256-dim via FluidAudio (raw cosine similarity) |
| Emotion/Event | SenseVoice-Small int8 via sherpa-onnx-go-macos (cgo) |
| Noise Reduction | DeepFilterNet3 via `metr-denoise` Swift CLI (MLX Metal GPU) |
| Audio Conversion | ffmpeg (embedded binary, auto clipping protection) |
| Audio I/O | go-audio/wav |
| Binary Packaging | embed.FS (no build tags) |

## Project Structure

```
cmd/commands/          CLI commands (transcribe, enroll, speakers, init, pack, unpack)
embedded/              embed.FS binary packaging + extraction
internal/
├── asr/               whisper-cli wrapper + SRT parser + content-based cache
├── audio/             ffmpeg conversion + WAV I/O + clipping detection
├── config/            Viper config loader + template generator
├── diarize/           metr-diarize subprocess wrapper (FluidAudio)
├── emotion/           SenseVoice classifier (sherpa-onnx)
├── models/            Model registry + auto-download + archive extraction
├── output/            TXT (with punctuation) / JSON / SRT formatters
├── punctuation/       CT-Transformer punctuator (sherpa-onnx)
├── speaker/           WeSpeaker store + auto-enroll + profile management
└── types/             Core data types + emotion mappings
tools/
├── metr-diarize/      Swift 6.0 CLI for FluidAudio diarization + voiceprint
└── metr-denoise/      Swift CLI for DeepFilterNet3 noise reduction
docs/                  SPEC + design docs + implementation plans
```

## License

TBD
