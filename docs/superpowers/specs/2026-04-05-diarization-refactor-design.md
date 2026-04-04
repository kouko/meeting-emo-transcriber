# Speaker Diarization 重構設計規格

> 用 sherpa-onnx OfflineSpeakerDiarization 取代逐段 embedding 比對，實現業界標準的 speaker diarization pipeline。

## 1. 背景與問題

### 問題

Phase 3/4 的 speaker identification 是逐段線性判定：每個 ASR segment（平均 3.8 秒）獨立提取 embedding → 比對。實測結果：4 人會議被辨識為 296 個 speaker。

根本原因：
1. ASR segment 太短（3.8 秒平均），embedding 不穩定
2. 逐段判定無全局視角，無法有效聚類
3. 不符合業界標準做法（clustering-based diarization）

### 業界標準做法

```
管線 A: 音訊 → ASR → 「說了什麼」（文字 + 時間戳）
管線 B: 音訊 → Diarization → 「誰在說話」（speaker ID + 時間段）
合併: 時間重疊對齊 → 「誰說了什麼」
```

ASR 和 Diarization 是**獨立的子系統**，最後才合併。WhisperX、MacWhisper、whisper-diarization 都用這個架構。

### 目標

用 sherpa-onnx `OfflineSpeakerDiarization` 實現標準 diarization pipeline：
- Pyannote segmentation model（專門為 speaker 切段設計）
- Speaker embedding extraction（全局）
- FastClustering（一次性聚類所有 segments）

## 2. 新架構

### Transcribe 流程

```
1. Extract binaries + ensure models
2. Convert audio → 16kHz mono WAV → read samples
3. [管線 A] Whisper ASR → []ASRResult（文字 + 時間戳）
4. [管線 B] sherpa-onnx OfflineSpeakerDiarization
   → Pyannote segmentation → speaker embedding → FastClustering
   → []DiarSegment（speaker_0/1/2... + 時間段）
5. 合併: 每個 ASR segment 找時間重疊最多的 diarization segment → 標上 speaker ID
6. 真名對應: 每個 cluster 取代表音訊 → Extractor embedding → 與 enrolled profiles 比對
   - 匹配 → "Alice"
   - 不匹配 → 保持 "speaker_1"
   - 自動建立 speakers/speaker_N/ 資料夾 + .profile.json
7. Emotion classification（不變）
8. 組裝 TranscriptResult + 輸出
```

### 架構圖

```
                    ┌─────────────────────────────┐
                    │        Input Audio           │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    Convert → 16kHz WAV       │
                    └──────────┬──────────────────┘
                               │
                ┌──────────────┼──────────────────┐
                │              │                   │
      ┌─────────▼────┐  ┌─────▼──────────┐  ┌────▼────────┐
      │  Whisper ASR  │  │  Diarization   │  │  Emotion    │
      │  → text +     │  │  → speaker ID  │  │  (per seg)  │
      │    timestamps  │  │    + time      │  │             │
      └─────────┬────┘  └─────┬──────────┘  └────┬────────┘
                │              │                   │
                └──────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  Merge: ASR × Diarization    │
                    │  + Enroll Profile Matching    │
                    │  → Speaker Names             │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │     TranscriptResult         │
                    │     → TXT / JSON / SRT       │
                    └─────────────────────────────┘
```

## 3. 新模組：`internal/diarize/`

### 目錄結構

```
internal/diarize/
├── diarize.go        # Diarizer wrapper (sherpa-onnx OfflineSpeakerDiarization)
├── merge.go          # ASR + Diarization 合併 + enrolled profile 真名對應
└── diarize_test.go   # 合併邏輯測試
```

### diarize.go — Diarizer

```go
package diarize

// Segment represents a diarization result.
type Segment struct {
    Start   float64
    End     float64
    Speaker int // 0-indexed cluster ID
}

// Diarizer wraps sherpa-onnx OfflineSpeakerDiarization.
type Diarizer struct { ... }

// NewDiarizer creates a diarizer with Pyannote segmentation + embedding models.
// If numClusters > 0, uses fixed cluster count; otherwise uses threshold.
func NewDiarizer(segModelDir, embModelPath string, threads int, numClusters int, threshold float32) (*Diarizer, error)

// Process runs diarization on audio samples (16kHz mono float32).
// Returns segments sorted by start time.
func (d *Diarizer) Process(samples []float32) []Segment

// Close releases resources.
func (d *Diarizer) Close()
```

### merge.go — 合併 + 真名對應

```go
// AssignSpeakers maps ASR results to diarization speakers by maximum time overlap.
// Returns speaker labels (e.g., "speaker_0", "speaker_1") for each ASR result.
func AssignSpeakers(asrResults []types.ASRResult, diarSegments []Segment) []int

// ResolveSpeakerNames maps cluster IDs to enrolled speaker names.
// For each cluster, extracts representative audio, computes embedding via Extractor,
// and matches against enrolled profiles.
// Unmatched clusters get "speaker_N" names and auto-create directories.
func ResolveSpeakerNames(
    clusterIDs []int,
    diarSegments []Segment,
    wavSamples []float32,
    sampleRate int,
    extractor *speaker.Extractor,
    profiles []types.SpeakerProfile,
    matcher *speaker.Matcher,
    threshold float32,
    store *speaker.Store,
) []string
```

ResolveSpeakerNames 邏輯：
1. 對每個 cluster ID，從 diarSegments 中找該 speaker 的所有 segments
2. 選最長的 segment（或合併多個 segment）作為代表音訊
3. 用 Extractor 提取 embedding
4. 用 Matcher 與 enrolled profiles 比對
5. 匹配 → 用 enrolled name（"Alice"）
6. 不匹配 → "speaker_N"，自動建立 `speakers/speaker_N/` + 儲存代表音訊 + `.profile.json`

## 4. 模型需求

### 新增模型

| 模型 | 用途 | 來源 | 大小 | 格式 |
|------|------|------|------|------|
| `sherpa-onnx-pyannote-segmentation-3-0` | Speaker segmentation | sherpa-onnx releases (tar.bz2) | ~5MB | ONNX (archive) |
| `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | Speaker embedding (for diarization) | sherpa-onnx releases | ~50MB | ONNX |

### 保留模型

| 模型 | 用途 | 備註 |
|------|------|------|
| `campplus-sv-zh-cn` | Enroll + verify 的 embedding | 保留給 enroll/verify 命令使用 |

### 為什麼 diarization 用 ERes2Net 而非 CAM++

sherpa-onnx 的 diarization 範例使用 `eres2net_base`。兩者都是 3DSpeaker 模型，但 eres2net 在 diarization 場景下可能更穩定。CAM++ 保留給 enroll/verify（已有 profiles 用 CAM++ 建立）。

## 5. 影響範圍

### 新增

| 檔案 | 用途 |
|------|------|
| `internal/diarize/diarize.go` | Diarizer wrapper |
| `internal/diarize/merge.go` | 合併 + 真名對應 |
| `internal/diarize/diarize_test.go` | 測試 |

### 修改

| 檔案 | 變更 |
|------|------|
| `internal/models/registry.go` | 新增 segmentation + eres2net 模型 |
| `internal/models/manager.go` | 支援 segmentation model 的 tar.bz2 下載（已有） |
| `cmd/commands/transcribe.go` | 重構：加入 diarization pipeline + 移除 discovery |

### 刪除

| 檔案 | 原因 |
|------|------|
| `internal/speaker/discovery.go` | 被 diarize 模組取代 |
| `internal/speaker/discovery_test.go` | 對應刪除 |

### 保留（不變）

| 檔案 | 原因 |
|------|------|
| `internal/speaker/extractor.go` | enroll + verify + cluster→profile 比對 |
| `internal/speaker/matcher.go` | 同上 |
| `internal/speaker/store.go` | Profile 管理 |
| `cmd/commands/enroll.go` | 不變 |
| `cmd/commands/speakers.go` | 不變 |
| `internal/emotion/classifier.go` | 不變 |

### CLI 變更

| Flag | 變更 |
|------|------|
| `--no-discover` | **移除**（diarization 永遠自動分群） |
| `--threshold` | 保留，但語意改為 diarization clustering threshold |
| `--num-speakers` | **新增**（可選，指定預知的說話者人數） |

## 6. 驗證計畫

### 單元測試

| 模組 | 測試內容 |
|------|---------|
| `diarize` | AssignSpeakers: ASR segment 對齊到正確的 diarization speaker |
| `diarize` | AssignSpeakers: 邊界情況（無重疊、完全包含、部分重疊） |
| `diarize` | ResolveSpeakerNames: enrolled speaker 匹配 |
| `diarize` | ResolveSpeakerNames: 未匹配 → speaker_N + 自動建資料夾 |

### 整合測試

1. **Diarization E2E**：用多人音檔測試，確認 speaker 數量合理
2. **真名對應 E2E**：enroll → transcribe → 確認已知 speaker 顯示真名
3. **自動建資料夾 E2E**：未知 speaker 自動建立 speakers/speaker_N/
4. **Emotion 不受影響**：確認 emotion 欄位仍正常

### 手動驗證

- 用 `/Users/kouko/Downloads/20260320 1401 Recording.mp3`（4 人會議）測試
- 目標：辨識出 ~4 個 speaker（而非 296 個）
