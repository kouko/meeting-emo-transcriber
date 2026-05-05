# Speaker Identification 4-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 speaker-identification improvements per `docs/superpowers/specs/2026-05-05-speaker-id-improvements-design.md`: (1) EER threshold auto-calibration, (2) filter + mean-pool centroid recompute, (3) high-confidence continuous enrollment, (4) 3-tier confidence band output.

**Architecture:** Each fix is independent of the others (Fix 2 has zero deps; Fix 4/3 derive thresholds from Fix 1). Ship as **4 separate PRs** in order Fix 2 → Fix 1 → Fix 4 → Fix 3. No stacked PRs.

**Tech Stack:** Go 1.25 (no new cgo dependencies). All Fix 2 audio extraction reuses existing `metr-diarize --extract-voiceprints` batch mode (Swift CLI unchanged).

---

## File Structure

### New files

| File | Responsibility | Phase |
|------|----------------|-------|
| `internal/speaker/centroid.go` | Filter + mean-pool cluster centroid | 1 (Fix 2) |
| `internal/speaker/centroid_test.go` | Filter + mean-pool tests | 1 |
| `internal/speaker/calibrate.go` | EER calibration logic | 2 (Fix 1) |
| `internal/speaker/calibrate_test.go` | EER computation tests | 2 |
| `cmd/commands/calibrate.go` | `metr calibrate` subcommand | 2 |
| `internal/speaker/continuous.go` | Continuous enrollment + auto/ persistence | 4 (Fix 3) |
| `internal/speaker/continuous_test.go` | FIFO + drift detection tests | 4 |

### Modified files

| File | Changes | Phase |
|------|---------|-------|
| `internal/diarize/merge.go` | `ResolveSpeakerNames`: use filtered mean-pool centroid (Fix 2); 3-tier classification (Fix 4); continuous enrollment hook (Fix 3) | 1, 3, 4 |
| `internal/types/types.go` | `Confidence` struct: add `SpeakerBand` + `SpeakerCandidates`; new `Candidate` type | 3 (Fix 4) |
| `internal/speaker/enroll.go` | `AutoEnroll` recognizes `auto/` subdir | 4 (Fix 3) |
| `internal/speaker/store.go` | List + manage `auto/` subdir | 4 (Fix 3) |
| `internal/output/txt.go` | Optional `?` suffix for tentative; `--show-confidence` flag rendering | 3 (Fix 4) |
| `internal/output/json.go` | Output full Confidence struct | 3 (Fix 4) |
| `internal/output/srt.go` | Match TXT behavior (no suffix by default) | 3 (Fix 4) |
| `internal/config/config.go` | Add `calibrated_eer`, `min_centroid_segment_duration`, `continuous_enrollment*` fields | 1, 2, 4 |
| `cmd/commands/root.go` | Register `calibrate` subcommand | 2 |
| `cmd/commands/transcribe.go` | Pass new config values to `ResolveSpeakerNames` | 1, 3, 4 |
| `example.config.yaml` | Document all new fields | 1, 2, 3, 4 |
| `CHANGELOG.md` | Per-fix entry | 1, 2, 3, 4 |

### Deleted files

None.

---

## Phase 1 — Fix 2: Filter + mean-pool centroid

**Branch:** `feat/centroid-mean-pool`
**PR target:** main
**Estimated:** ~550 LOC, 3-4 commits, ~1 day

### Task 1.1: Add `CentroidFilter` config struct + segment filter logic

- [ ] Define `CentroidFilter` struct in `internal/speaker/centroid.go`:
  ```go
  type CentroidFilter struct {
      MinDuration           float64
      ExcludeOverlap        bool
      MinRMS                float64
      MaxSegmentsPerCluster int
  }
  ```
- [ ] Implement `FilterClusterSegments(segments []diarize.Segment, clusterID string, allSegments []diarize.Segment, samples []float32, sampleRate int, filter CentroidFilter) []diarize.Segment`:
  - Filter segments where `seg.Speaker == clusterID`
  - Drop segments with `seg.End - seg.Start < filter.MinDuration`
  - If `ExcludeOverlap`: drop segments that overlap any other cluster's segment in `allSegments`
  - Drop segments where `audio.RMS(samples[startIdx:endIdx]) < filter.MinRMS` (reuse existing `audio.RMS()` from PR #14)
  - Sort remaining by duration descending; take top `MaxSegmentsPerCluster`
- [ ] Edge cases: empty input → empty output; all overlap → empty; sample range out of bounds → skip with warning

### Task 1.2: Add WAV slicing helper

- [ ] Implement `WriteSegmentWavs(samples []float32, sampleRate int, segments []diarize.Segment) ([]string, func(), error)` in `internal/speaker/centroid.go`:
  - Write each segment to a temp WAV file in `os.TempDir()/metr-centroid-*/`
  - Return paths + cleanup function (deferred caller)
  - Handle `samples[startIdx:endIdx]` slicing carefully (sample-rate × time → index)
- [ ] Reuse audio package WAV-writing utility if exists; otherwise add one
- [ ] Test with 1 known segment: verify WAV file is 16-bit PCM mono 16kHz, correct duration

### Task 1.3: Add mean-pool helper

- [ ] Implement `MeanPool(vectors [][]float32) []float32`:
  - Element-wise mean across all vectors
  - Returns nil if input empty
- [ ] Test: mean-pool of 3 known vectors matches expected element-wise mean

### Task 1.4: Wire into `ResolveSpeakerNames`

- [ ] In `internal/diarize/merge.go ResolveSpeakerNames`, before calling `matchAgainstProfilesDetailed`:
  - Call `FilterClusterSegments` for the current `clusterID`
  - If `len(filtered) >= cfg.MinSegmentsForRecompute` (default 2):
    - Call `WriteSegmentWavs` → get temp paths
    - Call `diarize.ExtractVoiceprints(diarizeBinPath, paths)` (already exists in `internal/diarize/diarize.go`)
    - Mean-pool vectors → use as centroid
    - Defer cleanup
  - Else: use existing `diarResult.SpeakerVoiceprints[clusterID]` as fallback + log "fallback to FluidAudio centroid for cluster X (insufficient filtered segments)"
- [ ] Pass `CentroidFilter` and `MinSegmentsForRecompute` from config through `ResolveSpeakerNames` signature
- [ ] Update `cmd/commands/transcribe.go` to plumb new config values

### Task 1.5: Config + CLI flags

- [ ] Add fields to `internal/config/config.go`:
  - `MinCentroidSegmentDuration float64` (default 1.5)
  - `ExcludeOverlapInCentroid bool` (default true)
  - `MaxSegmentsPerCluster int` (default 10)
  - `MinSegmentsForRecompute int` (default 2)
- [ ] Add corresponding CLI flags (camelCase converted; `--min-centroid-segment-duration` etc.)
- [ ] Update `example.config.yaml` with comments

### Task 1.6: Unit tests

- [ ] `TestFilterClusterSegments_DurationFilter`: feed 5 segments (1 too short, 4 OK) → expect 4
- [ ] `TestFilterClusterSegments_OverlapFilter`: feed 3 segments where 1 overlaps another cluster → expect 2
- [ ] `TestFilterClusterSegments_RMSFilter`: feed 3 segments, 1 with all-zero samples → expect 2
- [ ] `TestFilterClusterSegments_TopK`: feed 15 segments → expect top-10 by duration
- [ ] `TestMeanPool_ThreeVectors`: known input → known output
- [ ] `TestWriteSegmentWavs_Cleanup`: cleanup function deletes temp dir

### Task 1.7: Integration test (manual / smoke)

- [ ] Run `go test ./...` on full module
- [ ] Run `metr <test-meeting.mp3>` against a known fixture (or any local audio); verify:
  - stderr logs "filtered N segments for cluster X" lines
  - matching results changed (or didn't, with explanation) vs. baseline
  - No temp files leaked (`ls /tmp/metr-centroid-*` is empty post-run)

### Task 1.8: Docs + commit

- [ ] CHANGELOG.md entry under "Unreleased": `### Changed: cluster centroid for identification recomputed from filtered segments`
- [ ] Commit pattern (TDD-style):
  1. `test: add CentroidFilter + FilterClusterSegments tests` (red)
  2. `feat: add CentroidFilter + FilterClusterSegments` (green)
  3. `feat: add WAV segment writing + mean-pool` (with tests)
  4. `feat: wire mean-pool centroid into ResolveSpeakerNames` (with config plumbing)

### Phase 1 self-review checklist

- [ ] No leaked temp files (`defer cleanup()` in all paths)
- [ ] Fallback path tested (cluster with no qualifying segments uses FluidAudio centroid)
- [ ] Performance: 30-cluster meeting overhead < 30s wall time
- [ ] Backward compat: default config produces same behavior as before if `MinSegmentsForRecompute` raised to 999 (escape hatch)

---

## Phase 2 — Fix 1: EER calibration

**Branch:** `feat/eer-calibration`
**PR target:** main (after Phase 1 merged)
**Estimated:** ~400 LOC, 1-2 commits, ~half day

### Task 2.1: Implement distribution computation

- [ ] In `internal/speaker/calibrate.go`:
  ```go
  func ComputeDistributions(profiles []types.SpeakerProfile) (genuine, impostor []float32, err error)
  ```
- [ ] Genuine: for each profile with N voiceprints, generate N×(N-1)/2 cosine sim values
- [ ] Impostor: for each pair of profiles, generate Ni × Nj cosine sim values
- [ ] Return error if any profile has < 2 voiceprints (insufficient genuine pairs); test cases must verify

### Task 2.2: Implement EER finder

- [ ] Implement `FindEER(genuine, impostor []float32) (threshold, eer float32)`:
  - Sort genuine ascending, impostor descending
  - Sweep candidate threshold values (e.g., 0.30 to 0.95 in 0.001 steps, OR use unique values from both arrays)
  - At each candidate: compute FRR = #genuine < threshold / len(genuine); FAR = #impostor >= threshold / len(impostor)
  - EER = first threshold where FAR ≤ FRR; linearly interpolate for sub-step precision
- [ ] Edge cases:
  - genuine fully separates from impostor → return min(genuine) - 0.05, EER = 0
  - completely overlapping → return median, EER = high (e.g., 0.5); add "model insufficient" warning

### Task 2.3: Implement calibration report

- [ ] `CalibrationReport(genuine, impostor []float32, threshold, eer float32) string`:
  - Print histogram (ASCII art) of both distributions
  - Print EER point + recommended threshold + tentative/confident derivatives
  - Print warnings if data is insufficient
- [ ] Format example:
  ```
  Genuine pairs:   89 (cosine 0.61 - 0.94, mean 0.78)
  Impostor pairs: 421 (cosine 0.12 - 0.62, mean 0.34)

  Distribution:
    Genuine:  [........░░░▒▒▓▓██▓▓▒░] 0.6 ─── 0.9
    Impostor: [▓▓██▓▓▒▒░░░......]      0.1 ─── 0.6

  EER: 0.058 at threshold 0.583
  → match_threshold:    0.58
  → tentative_threshold: 0.58 (= EER)
  → confident_threshold: 0.73 (= EER + 0.15)
  ```

### Task 2.4: Implement `metr calibrate` subcommand

- [ ] In `cmd/commands/calibrate.go`:
  ```go
  func calibrateCmd() *cobra.Command
  ```
- [ ] Flags:
  - `--apply` (write back to config.yaml)
  - `--target-far <float>` (use FAR target instead of EER)
  - `--output <path>` (write report to file)
  - `--speakers <path>` (override default vault path; existing convention)
- [ ] Behavior:
  1. Load profiles from store
  2. Run `ComputeDistributions` + `FindEER`
  3. Print report
  4. If `--apply`: update `_metr/config.yaml` with new `calibrated_eer` + `calibrated_at`
  5. Exit 0 on success; non-zero with explanation on insufficient data

### Task 2.5: Config integration

- [ ] Add fields to `internal/config/config.go`:
  - `CalibratedEER float32` (default 0; if 0, falls back to MatchThreshold)
  - `CalibrationEERValue float32` (the actual error rate at EER)
  - `CalibratedAt string` (timestamp)
  - `CalibratedWithNProfiles int`
- [ ] In `cmd/commands/transcribe.go`:
  - Resolve effective threshold: if `CalibratedEER > 0`, use it; else use `MatchThreshold`
  - Print one-line stderr if `CalibratedAt` is > 30 days old: "Vault may have changed; consider re-running `metr calibrate`."

### Task 2.6: Unit tests

- [ ] `TestComputeDistributions_KnownInput`: 3 speakers × 3 voiceprints each → expect 9 genuine + 27 impostor pairs
- [ ] `TestComputeDistributions_InsufficientData`: 1 speaker × 1 voiceprint → return error
- [ ] `TestFindEER_FullySeparated`: synthetic distributions with no overlap → EER = 0, threshold at gap
- [ ] `TestFindEER_PartialOverlap`: distributions that intersect → EER > 0, threshold somewhere reasonable
- [ ] `TestCalibrationReport_FormatStability`: golden file test for report format

### Task 2.7: Docs + commit

- [ ] CHANGELOG entry: `### Added: metr calibrate subcommand for EER threshold auto-calibration`
- [ ] README section: "Calibrating speaker matching threshold"
- [ ] example.config.yaml: document `calibrated_eer` and how it overrides `match_threshold`
- [ ] Commits:
  1. `test: EER computation tests`
  2. `feat: ComputeDistributions + FindEER`
  3. `feat: metr calibrate subcommand`

### Phase 2 self-review checklist

- [ ] EER value is reproducible (deterministic given same input)
- [ ] `--apply` is the only mutation path; bare `metr calibrate` is read-only
- [ ] Insufficient-data error message is actionable
- [ ] Report ASCII histograms render correctly in narrow terminals (80 cols)

---

## Phase 3 — Fix 4: 3-tier confidence band output

**Branch:** `feat/confidence-band-output`
**PR target:** main (after Phase 2 merged)
**Estimated:** ~250 LOC, 1 commit, ~half day

### Task 3.1: Extend types

- [ ] In `internal/types/types.go`:
  ```go
  type Candidate struct {
      Name       string  `json:"name"`
      Similarity float32 `json:"similarity"`
  }

  type Confidence struct {
      Speaker            float32     `json:"speaker"`
      SpeakerBand        string      `json:"speaker_band"`        // "confident" | "tentative" | "unknown"
      SpeakerCandidates  []Candidate `json:"speaker_candidates,omitempty"`
      Emotion            float32     `json:"emotion"`
  }
  ```
- [ ] Update all places that construct `Confidence{}` literals (likely in `merge.go` and possibly `transcribe.go`)
- [ ] Verify `omitempty` on `SpeakerCandidates` keeps JSON output clean for unknown band

### Task 3.2: Implement classification helper

- [ ] In `internal/speaker/matcher.go` (or new `internal/speaker/confidence.go`):
  ```go
  func ClassifyConfidence(sim, eerThreshold, confidentThreshold float32) string {
      switch {
      case sim >= confidentThreshold:
          return "confident"
      case sim >= eerThreshold:
          return "tentative"
      default:
          return "unknown"
      }
  }
  ```
- [ ] Add helper to extract top-K candidates from `matchResultInfo.Details`:
  ```go
  func TopKCandidates(details []matchDetail, k int) []types.Candidate
  ```

### Task 3.3: Wire into `ResolveSpeakerNames`

- [ ] Replace binary `if result.Matched / else` branching with 3-way classification:
  - Confident: existing matched path; populate `SpeakerBand: "confident"`
  - Tentative: NEW path — use `result.Name` (the best candidate); populate `SpeakerBand: "tentative"` + `SpeakerCandidates`; do NOT create speaker_N folder; do NOT trigger continuous enrollment (that's Phase 4)
  - Unknown: existing else path; populate `SpeakerBand: "unknown"`
- [ ] Plumb `confidentThreshold` from config (Phase 2 calibrated value + 0.15 margin) through function signature

### Task 3.4: Output format updates

- [ ] `internal/output/json.go`: ensure `Confidence` struct fully serialized including new fields (likely automatic)
- [ ] `internal/output/txt.go`:
  - Default: render speaker name unchanged regardless of band
  - If `--show-confidence` flag: append `[conf=0.62 tentative]` etc.
- [ ] `internal/output/srt.go`: match TXT default behavior

### Task 3.5: CLI flag

- [ ] Add `--show-confidence` flag in `cmd/commands/transcribe.go`
- [ ] Add `show_confidence_in_txt` config field (default false)

### Task 3.6: Unit tests

- [ ] `TestClassifyConfidence`: 3 inputs spanning all 3 bands → 3 expected outputs
- [ ] `TestTopKCandidates_Sort`: unsorted details → top 3 sorted by sim desc
- [ ] `TestTopKCandidates_KGreaterThanLen`: K=10 with 2 details → returns 2 items, no panic
- [ ] `TestOutputJSON_AllBands`: golden test with 3 segments (one per band) → expected JSON
- [ ] `TestOutputTXT_DefaultNoBand`: tentative band rendered without `?` by default
- [ ] `TestOutputTXT_ShowConfidence`: with flag, tentative rendered with `[conf=X.XX tentative]`

### Task 3.7: Docs + commit

- [ ] CHANGELOG: `### Added: 3-tier confidence band output (confident / tentative / unknown)`
- [ ] README: section explaining bands + when to use `--show-confidence`
- [ ] example.config.yaml: document new fields
- [ ] Single commit: `feat: 3-tier confidence band output for speaker identification`

### Phase 3 self-review checklist

- [ ] Tentative band does NOT create speaker_N folder (preserves vault hygiene)
- [ ] JSON additive change verified (old consumers still parse)
- [ ] TXT default unchanged (no `?` suffix unless flag)
- [ ] `omitempty` confirmed: unknown band doesn't bloat JSON

---

## Phase 4 — Fix 3: High-confidence continuous enrollment

**Branch:** `feat/continuous-enrollment`
**PR target:** main (after Phase 3 merged)
**Estimated:** ~350 LOC, 1-2 commits, ~half day

### Task 4.1: Implement continuous sample persistence

- [ ] In `internal/speaker/continuous.go`:
  ```go
  func PersistContinuousSample(
      store *Store,
      speakerName string,
      audioBytes []byte,    // WAV file bytes for the cluster's representative segment
      similarity float32,
      maxSamples int,
  ) error
  ```
- [ ] Behavior:
  - Create `~/metr-speakers/{name}/auto/` if missing
  - Write file as `{YYYYMMDD}_{HHMMSS}_sim{NNN}.wav` (NNN = sim × 100, e.g. `sim083`)
  - Compute hash, append to `KnownAudioHashes` to prevent AutoEnroll re-processing
  - Call `pruneAutoSamples(speakerName, maxSamples)` to enforce FIFO

### Task 4.2: Implement FIFO pruning

- [ ] `pruneAutoSamples(store *Store, speakerName string, max int) error`:
  - List `auto/*.wav` sorted by timestamp (parsed from filename or mtime)
  - Delete oldest until len ≤ max
  - Also remove corresponding entries from `KnownAudioHashes` if present (avoid stale hash leaks)

### Task 4.3: AutoEnroll recognizes `auto/`

- [ ] In `internal/speaker/store.go ListAudioFiles(name)`: include files from `auto/` subdir (just glob both `{name}/*.wav` and `{name}/auto/*.wav`)
- [ ] Test: `TestListAudioFiles_IncludesAutoSubdir`

### Task 4.4: Drift detection

- [ ] After `AutoEnroll` recomputes voiceprint, compare to previous voiceprint:
  ```go
  driftSim := CosineSimilarity(oldVoiceprint, newVoiceprint)
  if driftSim < cfg.DriftWarnThreshold {  // default 0.85
      log.Warnf("Profile %s drifted significantly (cosine %.2f vs previous)", name, driftSim)
  }
  ```
- [ ] If multiple voiceprints already, compare new to latest

### Task 4.5: Wire into `ResolveSpeakerNames`

- [ ] In matched + confident path (from Phase 3):
  - If `cfg.ContinuousEnrollment` enabled AND `result.BestSim > confidentThreshold`:
    - Pick longest segment from cluster (filter overlap if possible)
    - Extract WAV bytes (reuse `WriteSegmentWavs` helper from Phase 1, write to memory or temp file)
    - Call `PersistContinuousSample`
- [ ] Limit: at most 1 continuous sample per cluster per run (don't dump multiple from same cluster)

### Task 4.6: Config + CLI

- [ ] Add fields to `internal/config/config.go`:
  - `ContinuousEnrollment bool` (default true)
  - `ContinuousEnrollmentMaxSamples int` (default 10)
  - `ContinuousEnrollmentThresholdMargin float32` (default 0.15)
  - `DriftWarnThreshold float32` (default 0.85)
- [ ] Add CLI flag `--no-continuous-enrollment` (sets to false for one run; for privacy mode)
- [ ] Update example.config.yaml

### Task 4.7: Unit tests

- [ ] `TestPersistContinuousSample_NewFile`: writes correct path, FIFO ≤ N
- [ ] `TestPersistContinuousSample_FIFOPrune`: 11th sample dropping oldest
- [ ] `TestPersistContinuousSample_HashTracking`: hash appended to `KnownAudioHashes`
- [ ] `TestListAudioFiles_IncludesAutoSubdir`: `auto/` files returned
- [ ] `TestAutoEnroll_DriftDetection`: synthetic before/after voiceprint with low cosine sim → warning logged

### Task 4.8: Integration smoke test

- [ ] Run `metr` on a fixture twice; verify:
  - First run: high-conf matches don't add to `auto/` (initial enrollment was the source)
  - Wait, actually first run should: existing profiles will match the recording at high conf (because it WAS the enrollment audio); auto/ does grow
  - Second run: hash dedup prevents re-processing same audio
  - Drift not warned (same audio shouldn't drift)
- [ ] Manual: edit voiceprint slightly to simulate drift; verify warning fires

### Task 4.9: Docs + commit

- [ ] CHANGELOG: `### Added: continuous enrollment auto-grows speaker profiles from high-confidence matches`
- [ ] README: section on auto/ subdir, --no-continuous-enrollment flag, privacy implications
- [ ] example.config.yaml: document new fields with privacy warning

### Phase 4 self-review checklist

- [ ] Hash deduplication works (don't re-add same WAV next run)
- [ ] FIFO pruning verified (11th sample drops 1st)
- [ ] `--no-continuous-enrollment` honored
- [ ] Drift detection emits warning but doesn't block execution
- [ ] auto/ samples readable by AutoEnroll (already handled in Task 4.3)

---

## Cross-phase tasks

### Build + E2E verification

- [ ] After all 4 phases merged: full smoke run on a real meeting fixture
- [ ] Verify config file has all new fields after fresh init
- [ ] Verify backward compat: existing `~/metr-speakers/` from before this work still loads + matches

### Documentation

- [ ] README: update overview to mention 4 improvements + skill-wrapper plan link
- [ ] docs/superpowers/specs/2026-05-05-speaker-id-improvements-design.md: keep as design reference (don't modify post-spec)
- [ ] CHANGELOG: tag a release once all 4 phases ship (suggest v0.x.y bump)

### Open questions to resolve during implementation

1. **`metr-diarize --extract-voiceprints` behavior on 1.5s clips**: needs prototype test. If it rejects too-short input, may need to bump `MinDuration` to 2s+
2. **`min_centroid_segment_duration` empirical default**: spec proposes 1.5s; verify with real meeting fixture before locking
3. **Drift warning threshold (0.85)**: empirical; may need calibration command extension to derive per-vault
4. **Skill-wrapper interface contract**: `meeting-toolkit` (in monkey-skills repo) consumes the JSON output. Coordinate JSON schema before Phase 4 PR ships to avoid breaking change later

---

## Self-Review Checklist (final)

- [ ] All 4 phases implemented as 4 separate PRs
- [ ] No stacked PRs (per `feedback_stacked_pr_race.md`)
- [ ] Each PR independently mergeable + reverteable
- [ ] All TDD steps followed (test first when reasonable)
- [ ] CHANGELOG accurate per phase
- [ ] example.config.yaml comprehensive + commented
- [ ] No new cgo dependencies (Swift sidecar untouched)
- [ ] Backward compat: vault from pre-fix versions still works without migration
- [ ] Performance: 30-cluster meeting overhead < 30s wall time (Fix 2 main cost)
- [ ] Privacy: continuous enrollment opt-out tested + documented

---

## References

- Design spec: `docs/superpowers/specs/2026-05-05-speaker-id-improvements-design.md`
- Sister plan (binary's predecessor): `docs/superpowers/plans/2026-04-05-diarization-refactor.md`
- Skill wrapper (paired effort): `meeting-toolkit` plugin in `kouko/monkey-skills` (TBD design proposal)
