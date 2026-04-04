# Meeting Emotion Transcriber

macOS CLI tool (Go) for meeting transcription with speaker identification and emotion recognition.

Combines whisper.cpp (ASR), CAM++ (speaker embedding), and SenseVoice (emotion classification) into a single binary. All processing runs locally — no cloud uploads.

## Features

- **Speech-to-Text** — whisper.cpp with Metal GPU acceleration, supports zh-TW/zh/en/ja
- **Speaker Identification** — CAM++ 512-dim voiceprint matching with auto-discovery of unknown speakers
- **Emotion Recognition** — SenseVoice detects Happy/Sad/Angry/Neutral + audio events (Laughter, Applause, etc.)
- **Multiple Output Formats** — TXT (CC caption style), JSON (structured), SRT (subtitles)
- **Single Binary** — whisper-cli + ffmpeg embedded via `go:embed`, models auto-downloaded on first run
- **Portable Speaker Profiles** — copy `speakers/` folder to any Mac and it works

## Quick Start

```bash
# Build
git clone https://github.com/kouko/meeting-emo-transcriber.git
cd meeting-emo-transcriber
brew install ffmpeg
make all

# Setup
metr init

# Add speaker samples
mkdir -p speakers/Alice speakers/Bob
cp alice_sample.wav speakers/Alice/
cp bob_sample.wav speakers/Bob/

# Enroll speakers
metr enroll

# Transcribe
metr transcribe --input meeting.wav --format all
```

## Prerequisites

- macOS ARM64 (Apple Silicon)
- Go 1.25+
- Homebrew (`brew install ffmpeg`)

## Build

```bash
make all        # deps (whisper-cli + ffmpeg) + build
make build      # build only (if deps already prepared)
make test       # run all tests
make clean      # remove binary
make clean-all  # remove binary + deps + build cache
make info       # show platform and deps status
```

## CLI Usage

### Global Flags

```
--speakers <path>     Speaker directory (default: ./speakers)
--config <path>       Config file (default: <speakers-dir>/config.yaml)
--log-level <level>   debug|info|warn|error (default: info)
```

### `transcribe` — Transcribe a meeting recording

```bash
metr transcribe --input meeting.wav [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Audio file (WAV, MP3, M4A, FLAC, OGG, OPUS) |
| `--output` | *(auto)* | Output path (auto-appends format extension) |
| `--format` | `txt` | `txt`, `json`, `srt`, `all`, or comma-separated |
| `--language` | `auto` | `auto`, `zh-TW`, `zh`, `en`, `ja` |
| `--threshold` | `0.6` | Speaker similarity threshold (0.0-1.0) |
| `--no-discover` | `false` | Disable unknown speaker auto-discovery |

```bash
# Basic
metr transcribe --input meeting.wav

# All formats + Taiwanese Mandarin
metr transcribe --input meeting.m4a --format all --language zh-TW

# Strict matching, no auto-discovery
metr transcribe --input meeting.wav --threshold 0.75 --no-discover
```

### `enroll` — Register speaker voiceprints

```bash
metr enroll [--force]
```

Scans `speakers/` for audio samples, computes CAM++ embeddings, saves to `.profile.json`.

```
Scanning ./speakers/...
  Alice: 3 samples → embedding computed ✓ (created)
  Bob:   2 samples → unchanged (cached)
```

### `speakers list` — List registered speakers

```bash
metr speakers list
```

### `speakers verify` — Test speaker recognition accuracy

```bash
metr speakers verify --name Alice --audio test.wav
```

```
Verifying against Alice...
  Similarity: 0.89 (threshold: 0.60)
  Result: ✓ MATCH
```

### `init` — Initialize workspace

```bash
metr init
```

Creates `speakers/`, `output/`, and `speakers/config.yaml` template.

## Output Examples

### TXT

```
Alice [happily]
Today's quarterly data is very impressive.

Bob
Let me report the specific numbers.

speaker_1 [angrily]
Unit B's problems are still not resolved.
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
      "text": "Today's quarterly data is very impressive.",
      "confidence": {"speaker": 0.89, "emotion": 0.95}
    }
  ]
}
```

### SRT

```
1
00:00:00,000 --> 00:00:05,200
(Alice) [happily] Today's quarterly data is very impressive.

2
00:00:05,200 --> 00:00:12,000
(Bob) Let me report the specific numbers.
```

## Speaker Setup

```
speakers/
├── config.yaml              # Optional config overrides
├── Alice/
│   ├── office.wav           # Voice sample 1
│   ├── phone.m4a            # Voice sample 2
│   └── .profile.json        # Auto-generated (by enroll)
└── Bob/
    ├── recording.wav
    └── .profile.json
```

- Create a folder per speaker under `speakers/`
- Add 1+ audio samples (clear speech, 10-30 seconds each)
- Run `enroll` to compute embeddings
- Rename `speaker_N/` folders from auto-discovery to assign real names

## Configuration

`speakers/config.yaml` (all fields optional):

```yaml
language: "auto"              # auto|zh-TW|zh|en|ja
threshold: 0.6                # Speaker similarity (0.0-1.0)
format: "txt"                 # txt|json|srt|all
discover: true                # Auto-discover unknown speakers
threads: 8                    # CPU threads (default: core count)
strategy: "max_similarity"    # Matching strategy
```

## Architecture

```
transcribe --input meeting.mp3
    │
    ├── embedded.ExtractAll()          whisper-cli + ffmpeg
    ├── models.EnsureModel()           ASR + VAD + Speaker + Emotion models
    ├── audio.ConvertToWAV()           ffmpeg → 16kHz mono WAV
    ├── asr.Transcribe()               whisper-cli → SRT → []ASRResult
    │
    ├── For each segment:
    │   ├── speaker.Extract()          CAM++ → 512-dim embedding
    │   ├── speaker.Match()            Cosine similarity vs profiles
    │   ├── discovery.Identify()       Unknown → speaker_N auto-discovery
    │   └── emotion.Classify()         SenseVoice → emotion + event
    │
    └── output.Format()                TXT / JSON / SRT
```

## Project Structure

```
cmd/commands/          CLI commands (transcribe, enroll, speakers, init)
embedded/              go:embed binary packaging + extraction
internal/
├── asr/               whisper-cli wrapper + SRT parser
├── audio/             ffmpeg conversion + WAV I/O
├── config/            Viper config loader
├── emotion/           SenseVoice classifier (sherpa-onnx)
├── models/            Model registry + auto-download
├── output/            TXT / JSON / SRT formatters
├── speaker/           CAM++ extractor + matcher + store + discovery
└── types/             Core data types + emotion mappings
scripts/               Build scripts (whisper-cli, ffmpeg)
docs/                  SPEC + design docs + implementation plans
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Go 1.25 |
| ASR | whisper.cpp v1.7.3 (Metal GPU) |
| Speaker Embedding | CAM++ via sherpa-onnx |
| Emotion/Event | SenseVoice-Small int8 via sherpa-onnx |
| VAD | Silero VAD v6.2.0 |
| Audio I/O | go-audio/wav + ffmpeg |
| CLI | cobra + viper |
| Binary Packaging | go:embed (embed.FS) |

## License

TBD
