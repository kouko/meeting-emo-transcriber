# Meeting Emotion Transcriber

macOS CLI tool (Go) for meeting transcription with speaker identification and emotion recognition.

Combines whisper.cpp (ASR), FluidAudio (diarization), CAM++ (speaker embedding), and SenseVoice (emotion classification) into a single binary. All processing runs locally — no cloud uploads.

## Features

- **Speech-to-Text** — whisper.cpp with Metal GPU acceleration, supports zh-TW/zh/en/ja
- **Speaker Diarization** — FluidAudio `metr-diarize` Swift CLI with CoreML/ANE acceleration, two parallel pipelines merged by time overlap
- **Speaker Identification** — CAM++ 192-dim voiceprint matching against enrolled profiles
- **Emotion Recognition** — SenseVoice detects Happy/Sad/Angry/Neutral + audio events (Laughter, Applause, etc.)
- **Multiple Output Formats** — TXT (CC caption style), JSON (structured), SRT (subtitles)
- **Single Binary** — whisper-cli + ffmpeg + metr-diarize embedded via `embed.FS`, models auto-downloaded on first run
- **SRT Cache** — skips Whisper on repeated runs for the same audio file
- **Auto-Enroll** — `transcribe` automatically re-enrolls speakers if profiles have changed
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
- Swift 6.0+ (for building `metr-diarize`)
- Homebrew (`brew install ffmpeg`)

## Build

```bash
make all        # build Swift CLI (metr-diarize) + deps (whisper-cli + ffmpeg) + Go binary
make build      # build Go binary only (if deps already prepared)
make swift      # build metr-diarize Swift CLI only
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
| `--threshold` | `0.7` | Speaker similarity threshold (0.0-1.0) |
| `--num-speakers` | `0` | Expected number of speakers (0 = auto-detect) |

```bash
# Basic
metr transcribe --input meeting.wav

# All formats + Taiwanese Mandarin
metr transcribe --input meeting.m4a --format all --language zh-TW

# Strict matching with known speaker count
metr transcribe --input meeting.wav --threshold 0.75 --num-speakers 3
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
  Similarity: 0.89 (threshold: 0.70)
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
threshold: 0.7                # Speaker similarity (0.0-1.0)
format: "txt"                 # txt|json|srt|all
num_speakers: 0               # Expected speaker count (0 = auto)
threads: 8                    # CPU threads (default: core count)
strategy: "max_similarity"    # Matching strategy
```

## Architecture

Two parallel pipelines (ASR + diarization) merged by time overlap:

```
transcribe --input meeting.mp3
    │
    [1/8] embedded.ExtractAll()        whisper-cli + ffmpeg + metr-diarize
    [2/8] models.EnsureASR()           language-specific model (Breeze/Belle/Kotoba/Large-v3)
    [3/8] models.EnsureVAD()           silero-vad
    [4/8] audio.ConvertToWAV()         ffmpeg → 16kHz mono WAV
    │
    [5/8] ASR pipeline (with SRT cache):
    │       asr.Transcribe()           whisper-cli → SRT → []ASRResult
    │
    [6/8] Diarization pipeline (parallel):
    │       diarize.Run()              metr-diarize subprocess (FluidAudio CoreML/ANE)
    │                                  → []DiarSegment {start, end, speaker_id}
    │
    [7/8] speaker.Resolve()            merge ASR + diarization by time overlap
    │       CAM++ embedding → enrolled profile matching + auto-enroll
    │
    [8/8] output.Format()              TXT / JSON / SRT
```

## Project Structure

```
cmd/commands/          CLI commands (transcribe, enroll, speakers, init)
embedded/              embed.FS binary packaging + extraction
internal/
├── asr/               whisper-cli wrapper + SRT parser + SRT cache
├── audio/             ffmpeg conversion + WAV I/O
├── config/            Viper config loader
├── diarize/           metr-diarize subprocess wrapper (FluidAudio)
├── emotion/           SenseVoice classifier (sherpa-onnx)
├── models/            Model registry + auto-download
├── output/            TXT / JSON / SRT formatters
├── speaker/           CAM++ extractor + matcher + store
└── types/             Core data types + emotion mappings
scripts/               Build scripts (whisper-cli, ffmpeg)
tools/metr-diarize/    Swift 6.0 CLI for FluidAudio diarization
docs/                  SPEC + design docs + implementation plans
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Main CLI | Go 1.25 + cobra + viper |
| ASR | whisper.cpp v1.7.3 (Metal GPU, embedded binary) |
| Diarization | FluidAudio via `metr-diarize` Swift CLI (CoreML/ANE) |
| Speaker Embedding | CAM++ 192-dim via sherpa-onnx-go-macos (cgo) |
| Emotion/Event | SenseVoice-Small int8 via sherpa-onnx-go-macos (cgo) |
| VAD | Silero VAD v6.2.0 |
| Audio Conversion | ffmpeg (embedded binary) |
| Audio I/O | go-audio/wav |
| Binary Packaging | embed.FS (no build tags) |
| Swift CLI | Swift 6.0, FluidAudio library |

## License

TBD
