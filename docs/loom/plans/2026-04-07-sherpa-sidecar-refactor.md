# Sherpa-onnx sidecar refactor (Option C)

## Context

`metr` currently cgo-links `github.com/k2-fsa/sherpa-onnx-go-macos` directly
(via `internal/punctuation` and `internal/emotion`). That upstream module
hardcodes a build-time rpath pointing at its own module cache, so released
binaries embed a `/Users/runner/go/pkg/mod/...` path that doesn't exist on
user machines. The Homebrew flow fixes this by bundling the dylibs next to
the binary and rewriting rpath (`scripts/bundle-dylibs.sh`), but that means
`metr` is always shipped as a `bin/` + `lib/` pair — never a single file.

### Goal

Make `metr` a **truly portable single-file binary** that can be `curl`'d,
downloaded from a GitHub Release, or dropped on a USB stick without any
external dylibs. One file, double-click, it runs.

### Why a sidecar rather than static linking

- Upstream `sherpa-onnx-go-macos` only ships dylibs; static-linking requires
  forking the CMake pipeline for both `sherpa-onnx` and `onnxruntime`, which
  is a heavy ongoing maintenance burden (`onnxruntime` static builds are
  officially "experimental").
- The project **already has a proven pattern** for shipping native binaries
  inside `metr`: `whisper-cli`, `ffmpeg`, `metr-diarize`, `metr-denoise` are
  all bundled via `//go:embed` and extracted to `~/.metr/bin/` on first run
  (`embedded/embedded.go`). Adding a sherpa sidecar to this list reuses
  infrastructure and testing muscle that's already in place.
- dyld resolves a process's own dylib dependencies **before** `main()` runs,
  so a self-extracting loader inside `metr` can never work — but dyld for a
  *subprocess* runs with that subprocess's own rpath, so an extracted sidecar
  can happily find its dylibs at `@executable_path/.`.
- The main `metr` binary ends up with **zero cgo dependencies on sherpa**,
  so `go build` produces a pure-Go Mach-O that's immediately relocatable.

### Outcome

After this refactor:

1. `metr` is a single Mach-O file. `otool -L metr` lists only system libraries.
2. First run extracts `metr-sherpa` + two dylibs to `~/.metr/bin/` (same
   mechanism already used for whisper/ffmpeg/diarize/denoise).
3. `metr` spawns `metr-sherpa` once at the start of transcription, keeps it
   alive as a long-running child, and communicates over stdin/stdout.
4. Public APIs of `internal/punctuation` and `internal/emotion` are
   **unchanged** — callers in `cmd/commands/transcribe.go` and
   `internal/output/txt.go` don't see the refactor.
5. Homebrew formula can drop the `libexec/{bin,lib}` split and go back to
   `bin.install "metr"` (or we keep the tap as-is since it still works).

---

## Architecture

```
┌─────────────────────────────┐          stdin (gob)           ┌──────────────────────────┐
│  metr (pure Go, no cgo)     │ ─────────────────────────────▶ │  metr-sherpa (cgo)       │
│                             │                                │                          │
│  internal/punctuation       │                                │  sherpa.OfflinePunctuation
│  internal/emotion           │ ◀───────────────────────────── │  sherpa.OfflineRecognizer│
│                             │         stdout (gob)           │                          │
│  internal/sherpasidecar:    │                                │  cmd/metr-sherpa/main.go │
│    client.go (spawns child) │                                │                          │
│    protocol.go (shared)     │                                │  request/response loop   │
└─────────────────────────────┘                                └──────────────────────────┘
         │                                                              │
         │ embeds (go:embed all:bin)                                    │
         ▼                                                              │
    embedded/bin/darwin-arm64/                                          │
      ├── metr-sherpa                                                   │
      ├── libsherpa-onnx-c-api.dylib           ◀─── extracted to ───────┤
      ├── libonnxruntime.dylib                      ~/.metr/bin/ at     │
      ├── libonnxruntime.1.23.2.dylib               first run           │
      ├── whisper-cli  (unchanged)                                      │
      ├── ffmpeg       (unchanged)                                      │
      ├── metr-diarize (unchanged)                                      │
      └── metr-denoise (unchanged)
```

### Wire protocol: `encoding/gob`

Both sides are Go, so `encoding/gob` is the obvious choice over JSON:

- **Native `[]float32`**: emotion classification sends 8k–160k float samples
  per call (32 KB – 640 KB raw); base64-JSON would add ~33% overhead and
  burn CPU on encode/decode. gob writes raw float32 bytes with a small type
  header.
- **Zero dependencies** — stdlib only.
- **Stream-based** — one `gob.Encoder` + `gob.Decoder` per stdio pipe, used
  for the entire sidecar lifetime. No message framing code to maintain.
- **Type-safe** — change the protocol struct and the compiler tells you
  about every call site.

For future Linux/Windows support gob still works (same language on both
sides). If we ever need cross-language IPC, swap to Protobuf later — the
`internal/sherpasidecar` package is the only place that changes.

### Protocol shape

One request type per operation, one response type per operation, one
envelope that tags which operation the message is for. Requests carry an
`ID` so the client can match responses (paves the way for future pipelining
even though current usage is strictly synchronous).

```go
// internal/sherpasidecar/protocol.go
package sherpasidecar

type MsgKind uint8

const (
    MsgInit MsgKind = iota + 1  // Client → Server: set model dirs, thread count
    MsgInitAck                   // Server → Client: ready (or error)
    MsgPunctuate                 // Client → Server: add punctuation
    MsgPunctuateAck              // Server → Client: punctuated text
    MsgClassify                  // Client → Server: emotion + audio event
    MsgClassifyAck               // Server → Client: emotion result
    MsgShutdown                  // Client → Server: clean exit
)

type Envelope struct {
    Kind    MsgKind
    ID      uint64          // request id, echoed in the ack
    Init    *InitReq        // only set when Kind == MsgInit
    InitAck *InitAck
    Punct   *PunctReq
    PunctAck *PunctAck
    Class   *ClassReq
    ClassAck *ClassAck
}

type InitReq struct {
    PunctModelDir   string  // empty = don't load punctuator
    EmotionModelDir string  // empty = don't load classifier
    Threads         int
}

type InitAck struct {
    PunctReady   bool
    EmotionReady bool
    Error        string
}

type PunctReq struct {
    Text     string
    Language string
}

type PunctAck struct {
    Text  string
    Error string
}

type ClassReq struct {
    Samples    []float32
    SampleRate int
}

type ClassAck struct {
    Raw        string
    Label      string
    Display    string
    AudioEvent string
    Confidence float32
    Error      string
}
```

Using pointer fields inside the envelope keeps gob wire size tight (unused
fields encode as nil) and means there's exactly **one** type registered with
`gob.Register`, which simplifies setup.

### Lifecycle & error handling

- **Spawn point**: `cmd/commands/transcribe.go` — after `embedded.ExtractAll()`
  resolves all binary paths but before punctuation/emotion are needed.
- **One sidecar per transcription run**. Started lazily via a package-level
  singleton getter in `internal/sherpasidecar.Client` so tests that don't
  touch punctuation/emotion don't pay the spawn cost.
- **Graceful shutdown**: `defer client.Close()` at the top of `RunE` sends
  `MsgShutdown`, waits up to 2 s for the child to exit, then `SIGKILL`s.
- **Degradation modes** — preserve current behaviour:
  - Sidecar fails to spawn or `InitAck` reports `PunctReady=false`
    → `punctFunc` becomes a no-op, transcription continues, warning logged.
  - Sidecar dies mid-run (`io.EOF` from the decoder) → log warning, fall
    back to neutral emotion + no punctuation for remaining segments. **Do
    not try to respawn** — a crash once probably means a crash twice, and
    we don't want to mask real bugs.
  - Emotion init fails (current code treats this as fatal) → keep that
    behaviour. `InitAck.Error` is propagated to `transcribe.go` and aborts.
- **Concurrency**: current callers are synchronous, so the client wraps
  stdin writes + stdout reads in a single `sync.Mutex`. That's enough for
  today and simple enough to reason about.

### What the sidecar *doesn't* do

- **Model downloads stay in `metr`**. `models.EnsureModel()` still runs in
  the parent, resolves a local directory, and passes the absolute path to
  the sidecar in `InitReq`. This keeps network I/O, cache management, and
  progress reporting in the place that already owns them.
- **Audio extraction stays in `metr`**. `audio.ExtractSegment()` runs in
  the parent and sends raw `[]float32` over the pipe. The sidecar never
  touches `.wav` files.
- **Output formatting stays in `metr`**. The sidecar only returns raw
  strings; fullwidth-to-ASCII conversion, emotion label lookup, etc. stay
  in the existing wrapper packages so the sidecar's wire format is as
  minimal as possible.

---

## File-level changes

### New files

| Path | Purpose |
|---|---|
| `cmd/metr-sherpa/main.go` | Sidecar entry point. Reads gob envelopes on stdin, dispatches to in-memory `*sherpa.OfflinePunctuation` and `*sherpa.OfflineRecognizer`, writes acks on stdout. No CLI flags — config comes from `InitReq`. |
| `internal/sherpasidecar/protocol.go` | Shared gob types (imported by both main and sidecar). No cgo imports — pure Go. |
| `internal/sherpasidecar/client.go` | Client used by `metr` main process. `Spawn(binaryPath string) (*Client, error)` forks the sidecar, runs an exec.Cmd with Stdin/Stdout pipes, and exposes `Init`, `Punctuate`, `Classify`, `Close`. |
| `internal/sherpasidecar/server.go` | Request dispatch loop used by `cmd/metr-sherpa/main.go`. Imports `sherpa-onnx-go-macos`. Lives under `internal/` but is only compiled into the sidecar binary, not `metr` itself (see build separation note below). |
| `scripts/build-sherpa-sidecar.sh` | New build script that `go build`s `cmd/metr-sherpa`, then reuses the same rpath-rewrite + dylib-copy logic as `bundle-dylibs.sh`, and places the result at `embedded/bin/darwin-arm64/{metr-sherpa,libsherpa-onnx-c-api.dylib,libonnxruntime.dylib,libonnxruntime.*.dylib}`. |

### Modified files

| Path | Change |
|---|---|
| `internal/punctuation/punctuation.go` | Drop `sherpa` import. `Punctuator` now wraps a `*sherpasidecar.Client` + cached "is ready" flag. `AddPunct` sends `PunctReq` over the client. Public API (`NewPunctuator`, `AddPunct`, `Close`) **unchanged**. Constructor takes an extra `*sherpasidecar.Client` param. |
| `internal/emotion/classifier.go` | Drop `sherpa` import. Same pattern: wraps a client, same public API, constructor takes the client. |
| `cmd/commands/transcribe.go` | After `embedded.ExtractAll()` returns bin paths, spawn `sherpasidecar.Client`, call `Init` with both model dirs, pass the client into `punctuation.NewPunctuator` and `emotion.NewClassifier`. Defer `client.Close()`. |
| `embedded/embedded.go` | Extend `BinPaths` with `SherpaSidecar string` + extract the sidecar binary + dylibs via the existing `extractBinary` + new `extractDylib` helpers. `go:embed all:bin` already picks up new files so the directive itself doesn't change. |
| `Makefile` | New `build-sherpa-sidecar` target called from `build-deps`. Runs `scripts/build-sherpa-sidecar.sh`. |
| `.github/workflows/release.yml` | Remove the `Bundle sherpa-onnx dylibs and fix rpath` step (no longer needed — main binary has no sherpa dep). Tarball reverts to shipping a bare `metr` file. |
| `homebrew/Formula/metr.rb`, `scripts/update-formula.sh` | Revert `install` block back to `bin.install "metr"`. Single-file install. |
| `go.mod` | `github.com/k2-fsa/sherpa-onnx-go-macos` stays as a **direct** dependency (needed by `cmd/metr-sherpa`), but it's no longer reachable from `cmd/metr` so `go build ./cmd/metr` won't link it. |

### Files **not** touched

- `internal/models/` — model download/cache logic unchanged.
- `internal/audio/` — segment extraction unchanged.
- `internal/output/txt.go` — still calls `punctFunc(text, lang)` via the closure pattern; the closure now reaches into the sidecar-backed punctuator.
- `internal/types/emotion.go` — `LookupEmotion` and the SenseVoice map remain in the main binary (sidecar returns the raw tag, main does the mapping).
- Tests for `internal/punctuation` and `internal/emotion` — update them to use a fake client, but the test structure stays.

### Build separation note

The critical invariant: **nothing imported transitively from `cmd/metr` may import `sherpa-onnx-go-macos`**. If it does, the main binary picks up the cgo dependency again and we're back where we started.

Enforcement:

1. `internal/sherpasidecar/server.go` imports sherpa. It is **only** imported by `cmd/metr-sherpa/main.go`, never by anything under `cmd/metr` or `internal/punctuation` or `internal/emotion`.
2. Add a CI check (or a `go test` in `cmd/metr/...`) that asserts `go list -deps ./cmd/metr/...` does not contain `k2-fsa/sherpa-onnx-go-macos`. This is the single most important guardrail — without it, a stray import six months from now silently reintroduces the bug.

Sketch of the guardrail test:

```go
// cmd/metr/nodeps_test.go
func TestMetrBinaryHasNoSherpaDep(t *testing.T) {
    out, err := exec.Command("go", "list", "-deps", "./cmd/metr/...").Output()
    if err != nil { t.Fatal(err) }
    if bytes.Contains(out, []byte("sherpa-onnx-go-macos")) {
        t.Fatalf("metr main binary transitively depends on sherpa-onnx — sidecar invariant violated")
    }
}
```

---

## Build flow

```
make deps
  ├─ scripts/build-whisper.sh            (unchanged)
  ├─ scripts/build-diarize.sh            (unchanged)
  ├─ scripts/build-denoise.sh            (unchanged)
  ├─ scripts/build-ffmpeg.sh             (unchanged)
  └─ scripts/build-sherpa-sidecar.sh     (NEW)
       ├─ go build -o metr-sherpa ./cmd/metr-sherpa
       ├─ locate sherpa-onnx-go-macos via `go list -m`
       ├─ copy dylibs + fix rpath to @executable_path/.
       │   (rpath points at same dir, not ../lib, because under
       │    ~/.metr/bin/ the sidecar and dylibs sit side by side)
       ├─ ad-hoc codesign the rewritten sidecar
       ├─ smoke test: echo a ping envelope through stdin, verify ack
       └─ install to embedded/bin/darwin-arm64/

make build
  └─ go build -ldflags "..." -o metr ./cmd/metr
       (main binary, now pure Go, embeds everything via //go:embed)
```

**Critical subtlety**: the sidecar's rpath must be `@executable_path/.` not
`@executable_path/../lib`, because `embedded.ExtractAll()` flattens every
embedded file into the same directory (`~/.metr/bin/`). The sidecar and its
dylibs will end up as peers, not in separate `bin/` + `lib/` subdirs.
`bundle-dylibs.sh` cannot be reused verbatim — `build-sherpa-sidecar.sh`
should parameterise the rpath value.

Consider extracting the common rpath-rewrite logic into a shared helper
(e.g. `scripts/lib/fix-rpath.sh` sourced by both scripts) to avoid drift.

---

## Verification

### Unit tests

1. **`internal/sherpasidecar` round-trip test**: spawn the real sidecar
   binary against a tiny mock model (or use a test tag that makes the
   sidecar return canned responses without loading sherpa), send each
   message kind, assert acks match.
2. **`internal/punctuation` + `internal/emotion`**: swap the real client
   for a fake implementing a `Client` interface. Existing behavioural
   tests (fullwidth-to-ASCII, emotion map lookup) keep working.
3. **Guardrail**: the `TestMetrBinaryHasNoSherpaDep` test described above.

### Integration tests

1. `make deps && make build && ./metr --version` — must succeed on a clean
   machine with no dylibs anywhere in `DYLD_LIBRARY_PATH`.
2. `otool -L metr | grep -iE 'sherpa|onnx'` — **must be empty**. If this
   matches anything the refactor is incomplete.
3. `otool -l metr | grep LC_RPATH` — should show no `/Users/runner` or
   `/pkg/mod/` paths. Ideally no rpath at all, since main has no external
   dylibs.
4. End-to-end transcribe a 30-second test WAV — compare output against a
   golden file produced by the pre-refactor binary. Punctuation and
   emotion results must match byte-for-byte.
5. **Relocatability test**: `cp metr /tmp/anywhere/ && /tmp/anywhere/metr
   transcribe test.wav`. Must work. First run extracts to `~/.metr/bin/`
   and subsequent runs reuse the cache.
6. **Homebrew test**: after tagging a test release, `brew install
   kouko/tap/metr` on a fresh machine and run end-to-end. The formula
   install block is now trivial (`bin.install "metr"`), which is the
   whole point of the refactor.

### Performance regression check

Measure end-to-end transcription time on a known 5-minute meeting WAV
before and after the refactor. The IPC overhead budget per call:

- Punctuation: <1 ms per call, ~5–20 calls per meeting → **imperceptible**.
- Emotion: <5 ms per call (most of the time is gob-encoding ~200 KB of
  float32 samples + the pipe round trip), ~100–1000 calls per meeting →
  **0.5–5 s total overhead** in the worst case.

If the emotion overhead exceeds ~5 s on realistic inputs, measure whether
the bottleneck is gob encoding, pipe buffer sizing, or context switching,
and consider: (a) batching multiple segments into one `ClassReq`, or (b)
switching to a framed raw-binary protocol. Don't optimise prematurely —
ship the gob version first and measure.

### Failure mode tests

1. Delete `metr-sherpa` from `~/.metr/bin/` between runs → main should
   re-extract from the embedded FS.
2. `kill -9` the sidecar during a transcription → main should log a
   warning and finish the remaining segments with neutral emotion.
3. Run `metr` with `MET_CACHE_DIR=/tmp/metr-readonly` where the dir is
   chmod 555 → should fail fast with a clear error about cache write.

---

## Risks and open questions

1. **Gob stream resilience**: if a single malformed message corrupts the
   decoder state, the whole stream is dead. Mitigation: any decode error
   on the main side triggers sidecar restart-free degradation (no more
   punctuation/emotion for the rest of the run) rather than trying to
   resync. This matches the current graceful-degradation behaviour.

2. **Sidecar startup latency**: `sherpa.NewOfflineRecognizer` on the
   SenseVoice model is not instant (loads a ~230 MB model). Measure it.
   If >2 s, consider starting the sidecar in a goroutine at the top of
   `RunE` so model load overlaps with ASR, which is the first slow step.

3. **Pipe back-pressure**: a 640 KB emotion request writes to a pipe whose
   default buffer is ~64 KB on macOS. The sidecar must be actively reading
   while the client writes, otherwise the client blocks. Since the
   protocol is strictly request→response, this is fine — the sidecar's
   read loop is always the next thing running — but document it and don't
   accidentally introduce pipelining without handling back-pressure.

4. **Windows / Linux future**: gob is portable, but the build script is
   macOS-specific (`install_name_tool`, `otool`). When Linux support
   arrives, write a parallel `build-sherpa-sidecar-linux.sh` that handles
   ELF RPATH instead (`patchelf --set-rpath '$ORIGIN'`). Protocol code
   doesn't change.

5. **Model sharing across runs**: current design loads models fresh per
   sidecar spawn. If in future `metr` grows a "watch mode" or long-running
   daemon, the sidecar can stay alive across transcriptions for free —
   this refactor actually makes that easier.

6. **Sidecar binary size**: the sherpa sidecar will be ~8–10 MB plus the
   ~40 MB of dylibs. Embedded inside `metr` via `go:embed`, that inflates
   the main binary from ~11 MB to ~60 MB. Acceptable: still smaller than
   the existing `whisper-cli` + `ffmpeg` embed footprint, and it buys us
   the single-file property.

---

## Out of scope for this refactor

- Cross-platform support (Linux/Windows). Current build is darwin-arm64
  only; keep that scope but structure code to make the later platform
  split a separate build script, not a protocol change.
- Migrating `whisper-cli`, `ffmpeg`, `metr-diarize`, `metr-denoise` to the
  same sidecar pattern. They already work because they're standalone CLIs
  with their own dylib handling; don't disturb them.
- Notarization / Gatekeeper hardened runtime. Worth doing separately once
  the main binary is cgo-free — it'll be dramatically easier to notarize.
- Replacing gob with Protobuf. Only consider if cross-language sidecars
  become a requirement.

---

## Critical files to read before starting

- `cmd/commands/transcribe.go:151-273` — the current punctuation + emotion
  orchestration, which is what the refactor rewires.
- `embedded/embedded.go` — the extraction pattern the sidecar joins.
- `internal/punctuation/punctuation.go` and `internal/emotion/classifier.go`
  — tiny files; understand the current public API since that's what stays.
- `scripts/bundle-dylibs.sh` — the rpath-rewrite + codesign recipe that
  the new `build-sherpa-sidecar.sh` adapts.
- `Makefile` — to see how existing `build-deps` targets look so the new
  one fits in.

## Suggested task breakdown

Independent enough to parallelise if you want; numbered in execution order
otherwise:

1. Create `internal/sherpasidecar/protocol.go` with the gob types.
2. Create `cmd/metr-sherpa/main.go` + `internal/sherpasidecar/server.go`
   (pulls all sherpa calls out of `punctuation` and `emotion`). Verify in
   isolation by `go run ./cmd/metr-sherpa` and pushing canned gob envelopes
   at it.
3. Create `internal/sherpasidecar/client.go` with `Spawn`, `Init`,
   `Punctuate`, `Classify`, `Close`. Unit-test against the real binary.
4. Refactor `internal/punctuation/punctuation.go` and
   `internal/emotion/classifier.go` to use the client. Keep public API.
5. Wire `cmd/commands/transcribe.go` — spawn sidecar after
   `ExtractAll`, pass client into the wrappers, defer Close.
6. Write `scripts/build-sherpa-sidecar.sh` (factor shared rpath-rewrite
   code out of `bundle-dylibs.sh` into `scripts/lib/fix-rpath.sh`).
7. Extend `embedded/embedded.go` to extract the sidecar + dylibs. Update
   `BinPaths`.
8. Add `make build-sherpa-sidecar` and wire it into `deps`.
9. Add the `TestMetrBinaryHasNoSherpaDep` guardrail.
10. Simplify `.github/workflows/release.yml`, `homebrew/Formula/metr.rb`,
    `scripts/update-formula.sh` back to single-file install.
11. Run the full verification suite above, especially the relocatability
    + `otool -L` checks.
12. Tag a release, `brew upgrade`, end-to-end test on a clean machine.
