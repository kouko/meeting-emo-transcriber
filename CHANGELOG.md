# Changelog

## Unreleased — Speaker identification overhaul

Branch `claude/review-project-D9mUQ` lands a multi-stage rework of how
enrolled speakers are matched to diarization clusters. Motivated by a
research review of pyannote / NeMo / WeSpeaker production patterns and
the existing code's correctness gaps.

### Behaviour changes (defaults)

- `--match-threshold` default raised from **0.55 → 0.65**. WeSpeaker
  embeddings have ~0.65–0.75 EER on cosine, so 0.55 was permissive and
  produced false matches on similar-sounding speakers.
- New `--match-margin` flag (default **0.07**). A cluster only matches
  an enrolled name when its top cosine beats the runner-up by at least
  this gap. Smaller gaps are treated as ambiguous and the cluster is
  reassigned to a fresh `speaker_N` instead.
- Cluster → name assignment is now **one-to-one** (Hungarian-style
  greedy). Two clusters can no longer share the same enrolled name.
- Enrollment refuses to compute a voiceprint when **total clean speech
  is below 15s** (`EnrollMinDurationSec`).
- Learn-mode (`--learning-mode` / `-L`) review samples are now written
  under `<speakers>/_metr/review/` instead of at the root, so they no
  longer pollute the enrolled-speaker namespace on subsequent runs.

### New features

- **`metr speakers inspect <name>`** — diagnostic command that
  re-extracts embeddings from a speaker's enrolled audio files and
  reports intra-class cosine (per-file vs the merged voiceprint),
  inter-class cosine (vs other enrolled speakers), and a safety margin.
  Useful for tuning thresholds without running a full transcribe.
- **`metr <audio> --dry-run`** — runs only diarize + speaker matching,
  skipping ASR, emotion classification, and output writing. Lets you
  iterate on `--match-threshold` / `--match-margin` quickly because ASR
  is the slowest step.
- **`--verify-segments`** (opt-in) — after cluster-level matching,
  re-extracts an embedding for each ASR segment and demotes individual
  segments whose cosine falls below `--verify-threshold` (default 0.50)
  to `Unknown`. Catches stray speech that diarization merged into the
  wrong cluster.

### Algorithm changes

- **AutoEnroll** no longer concatenates all enrolled audio before
  extracting one embedding. It now extracts one embedding per file and
  L2-normalises + averages on the unit hypersphere (the canonical
  WeSpeaker / ECAPA / Kaldi recipe; concatenation is OOD for the
  statistics-pooling layers these models use).
- `Store.List()` now skips directories starting with `_`, formalising
  `_metr/` (and any future reserved prefixes) as non-enrolled
  namespaces.
- `Store.LoadProfile` now dedups `KnownAudioHashes` when merging
  multiple `*.profile.json` files.
- `CosineSimilarity` returns 0 (instead of panicking) on
  length-mismatched / empty input. Per-profile scoring is centralised
  through a new `speaker.BestSimilarity(emb, profile)` helper used by
  the matcher, cluster resolver, and segment re-verifier.

### Tests

- 50+ new unit tests across `internal/speaker/` and `internal/diarize/`
  covering: Hungarian assignment, margin guard, learn-mode isolation,
  enrollment duration gate, per-segment verification (8 test cases),
  inspect stats (10 test cases), and `BestSimilarity` edge cases.
- The new tests act as regression guards for the behavioural changes.

### Known follow-ups

These were flagged in the research review but not shipped yet — they
require either real-audio validation on macOS or a non-trivial new
dependency:

- **AS-Norm score normalisation.** WeSpeaker's official recipe achieves
  ~10–17% relative EER improvement with AS-Norm over raw cosine. Needs
  a ~6MB pre-computed impostor cohort.
- **Chinese WeSpeaker model swap.** The FluidAudio-shipped WeSpeaker
  ResNet34 is VoxCeleb-trained (English-dominant); CN-Celeb-trained
  variants (`resnet34_cnceleb`, `eres2net-cn-common-200k`,
  `campplus-cn-common-200k`) would substantially reduce EER on Mandarin
  meetings but need either CoreML conversion or a sherpa-onnx
  integration.
- **TS-VAD / Personal VAD.** Native target-speaker activity detection
  would beat the two-stage diarize-then-identify pipeline on overlap,
  but no production CoreML port exists today.
