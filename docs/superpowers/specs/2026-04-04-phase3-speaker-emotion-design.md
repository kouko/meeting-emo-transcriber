# Phase 3 設計規格：Speaker Embedding + Emotion Classification

> Phase 3 of meeting-emo-transcriber — 透過 sherpa-onnx Go API 整合 CAM++ 聲紋辨識和 SenseVoice 情感分類。

## 1. 背景與目標

### 為什麼做

Phase 2 完成了 ASR pipeline（whisper-cli → SRT → 逐字稿），但 transcribe 輸出中：
- Speaker 全部是 "Unknown"
- Emotion 全部是 "Neutral"
- AudioEvent 全部是 "Speech"

Phase 3 的目標是用 ONNX 模型推理填充這些欄位，讓輸出包含**真實的說話者辨識和情感標籤**。

### 預期成果

Phase 3 完成後：
- `enroll` 命令可以真正計算 speaker embedding（CAM++ ONNX）
- `transcribe` 輸出包含真實的 Speaker name、Emotion label、AudioEvent
- 模型首次使用時自動下載（CAM++ ~30MB + SenseVoice ~66MB）

### 範圍

| 包含 | 不包含 |
|------|--------|
| sherpa-onnx Go API 整合 | Unknown speaker auto-discovery（Phase 4） |
| CAM++ speaker embedding extraction | Pipeline 整合/分段策略（Phase 4） |
| SenseVoice emotion + audio event classification | Emotion2vec+ 替代模型 |
| enroll 命令完整實作 | speakers verify 命令 |
| transcribe 命令 speaker/emotion 填充 | |
| 移除 libonnxruntime.dylib 從 embed | |
| Model registry 新增 CAM++ + SenseVoice | |

---

## 2. 技術方案：sherpa-onnx Go API

### 為什麼選 sherpa-onnx

1. **CAM++ 的輸入需要 fbank 特徵提取**（80-dim mel filterbank），不是 raw waveform。用 `yalue/onnxruntime_go` 要從零實作特徵提取器。
2. sherpa-onnx 已經處理好所有音訊前處理（resampling、fbank、normalization）。
3. Go API 直接提供 `SpeakerEmbeddingExtractor`（CAM++）和 `OfflineRecognizer`（SenseVoice，回傳 Emotion + Event）。
4. 預編譯 macOS library 可用，不需自己編譯 C++ code。

### 依賴

**新增**：
```
github.com/k2-fsa/sherpa-onnx-go-macos  # macOS arm64 預編譯 sherpa-onnx + onnxruntime
```

**移除**：
- `embedded/binaries/darwin-arm64/libonnxruntime.dylib` 從 go:embed 移除
- sherpa-onnx 自帶 ONNX Runtime，不需要額外的 dylib

### Embedded 調整

```
BinPaths struct (Phase 3):
    WhisperCLI  string  // 保留
    FFmpeg      string  // 保留
    // ONNXRuntime 移除
```

修改清單：
- `internal/embedded/extract.go`：移除 libonnxruntime.dylib binSpec
- `internal/embedded/embed_prod.go`：移除 onnxruntime embed 宣告
- `internal/embedded/embedded.go`：BinPaths 移除 ONNXRuntime 欄位
- `internal/embedded/embedded_test.go`：更新測試
- `scripts/download-onnxruntime.sh`：保留但標為 optional
- `scripts/prepare-all.sh`：移除 onnxruntime 驗證

---

## 3. Speaker Embedding Extractor

### 目錄結構

```
internal/speaker/
├── store.go       # Phase 1（不變）
├── matcher.go     # Phase 1（不變）
├── extractor.go   # Phase 3 新增
└── extractor_test.go  # Phase 3 新增
```

### API

```go
package speaker

import sherpa "github.com/k2-fsa/sherpa-onnx-go-macos/sherpa_onnx"

// Extractor wraps sherpa-onnx SpeakerEmbeddingExtractor for CAM++ model.
type Extractor struct {
    inner *sherpa.SpeakerEmbeddingExtractor
    dim   int
}

// NewExtractor creates a speaker embedding extractor.
// modelPath: path to campplus_sv_zh-cn.onnx
func NewExtractor(modelPath string, threads int) (*Extractor, error)

// Extract computes a speaker embedding from audio samples.
// samples: float32 PCM at sampleRate Hz
// Returns: 512-dim float32 embedding vector
func (e *Extractor) Extract(samples []float32, sampleRate int) ([]float32, error)

// Dim returns the embedding dimension (512 for CAM++).
func (e *Extractor) Dim() int

// Close releases the underlying sherpa-onnx resources.
func (e *Extractor) Close()
```

### 實作流程

```
NewExtractor(modelPath, threads)
    └─ sherpa.NewSpeakerEmbeddingExtractor(config)

Extract(samples, sampleRate)
    ├─ stream := extractor.CreateStream()
    ├─ stream.AcceptWaveform(sampleRate, samples)
    ├─ extractor.IsReady(stream) → true
    ├─ embedding := extractor.Compute(stream)
    └─ return embedding ([]float32, len=512)
```

---

## 4. Emotion Classifier

### 目錄結構

```
internal/emotion/
├── classifier.go      # Phase 3 新增
└── classifier_test.go # Phase 3 新增
```

### API

```go
package emotion

import (
    sherpa "github.com/k2-fsa/sherpa-onnx-go-macos/sherpa_onnx"
    "github.com/kouko/meeting-emo-transcriber/internal/types"
)

// Classifier wraps sherpa-onnx OfflineRecognizer with SenseVoice model
// for emotion classification and audio event detection.
type Classifier struct {
    inner *sherpa.OfflineRecognizer
}

// NewClassifier creates an emotion classifier.
// modelPath: path to sensevoice-small-int8.onnx
func NewClassifier(modelPath string, threads int) (*Classifier, error)

// Classify performs emotion classification on audio samples.
// Returns: EmotionResult (with 3-layer mapping), audioEvent string, error
func (c *Classifier) Classify(samples []float32, sampleRate int) (types.EmotionResult, string, error)

// Close releases the underlying sherpa-onnx resources.
func (c *Classifier) Close()
```

### 實作流程

```
NewClassifier(modelPath, threads)
    └─ sherpa.NewOfflineRecognizer(config)
       config.ModelConfig.SenseVoice = OfflineSenseVoiceModelConfig{
           Model: modelPath,
           Language: "",  // auto-detect
           UseInverseTextNormalization: 0,
       }

Classify(samples, sampleRate)
    ├─ stream := recognizer.CreateStream()
    ├─ stream.AcceptWaveform(sampleRate, samples)
    ├─ recognizer.Decode(stream)
    ├─ result := recognizer.GetResult(stream)
    ├─ emotionRaw := result.Emotion    // "HAPPY", "SAD", "NEUTRAL", etc.
    ├─ audioEvent := result.Event      // "Speech", "Laughter", etc.
    ├─ emotionResult := types.LookupEmotion(emotionRaw, "sensevoice")
    └─ return emotionResult, audioEvent
```

### Emotion 三層轉換（使用 Phase 1 已有的 mapping）

sherpa-onnx 回傳的 `result.Emotion` → `types.SenseVoiceEmotionMap` 查表：
```
"HAPPY"   → {Label: "Happy",   Display: "happily"}
"SAD"     → {Label: "Sad",     Display: "sadly"}
"ANGRY"   → {Label: "Angry",   Display: "angrily"}
"NEUTRAL" → {Label: "Neutral", Display: ""}
"unk"     → {Label: "Unknown", Display: ""}
```

---

## 5. Model Registry 更新

### 新增模型

在 `internal/models/registry.go` 新增：

```go
"campplus-sv-zh-cn": {
    Name:     "campplus-sv-zh-cn",
    URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx",
    SHA256:   "",
    Size:     30000000,  // ~30MB
    Category: "speaker",
},
"sensevoice-small-int8": {
    Name:     "sensevoice-small-int8",
    URL:      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
    SHA256:   "",
    Size:     228000000,  // ~228MB (tar.bz2 archive, extract model.int8.onnx)
    Category: "emotion",
    // 注意：下載後需要解壓 tar.bz2，取出 model.int8.onnx + tokens.txt
},
```

### EnsureModel 擴展

`EnsureModel` 需要擴展：
1. 支援 `.onnx` 直接下載（CAM++ model）
2. 支援 `.tar.bz2` 下載 + 解壓（SenseVoice model archive）
   - 下載 archive → 解壓到 models 目錄 → 取出 `model.int8.onnx` + `tokens.txt`
   - SenseVoice 需要 tokens.txt 才能運作
3. 回傳的 path 指向解壓後的目錄（SenseVoice）或 `.onnx` 檔案（CAM++）

---

## 6. CLI 整合

### transcribe 命令修改

Phase 2 的 placeholder 替換為真實推理：

```go
// Phase 3: 初始化 ONNX 模型
extractor, _ := speaker.NewExtractor(speakerModelPath, cfg.Threads)
defer extractor.Close()

classifier, _ := emotion.NewClassifier(emotionModelPath, cfg.Threads)
defer classifier.Close()

// 載入 speaker profiles
store := speaker.NewStore(speakersDir, config.SupportedAudioExtensions())
profiles, _ := store.LoadProfiles()
matcher := speaker.NewMatcher(&speaker.MaxSimilarityStrategy{})

// 讀取完整 WAV（用於片段切割）
wavSamples, sampleRate, _ := audio.ReadWAV(wavPath)

// 每個 ASR segment
for _, asr := range asrResults {
    segAudio := audio.ExtractSegment(wavSamples, sampleRate, asr.Start, asr.End)

    // Speaker identification
    embedding, _ := extractor.Extract(segAudio, sampleRate)
    matchResult := matcher.Match(embedding, profiles, float32(cfg.Threshold))

    // Emotion classification
    emotionResult, audioEvent, _ := classifier.Classify(segAudio, sampleRate)

    segments = append(segments, types.TranscriptSegment{
        Start:      asr.Start,
        End:        asr.End,
        Speaker:    matchResult.Name,  // 真實 speaker name 或空字串
        Emotion:    types.EmotionInfo{Raw: emotionResult.Raw, Label: emotionResult.Label, Display: emotionResult.Display},
        AudioEvent: audioEvent,
        Language:   asr.Language,
        Text:       asr.Text,
        Confidence: types.Confidence{Speaker: matchResult.Similarity, Emotion: emotionResult.Confidence},
    })
}
```

未匹配的 speaker（`matchResult.Name == ""`）暫時標為 "Unknown"（Phase 4 實作 auto-discovery）。

### enroll 命令修改

Phase 1 stub 替換為真實 embedding 計算：

```go
// 初始化 extractor
extractor, _ := speaker.NewExtractor(speakerModelPath, cfg.Threads)
defer extractor.Close()

for _, name := range names {
    files, _ := store.ListAudioFiles(name)
    needsUpdate := force || store.NeedsUpdate(name)

    if !needsUpdate {
        fmt.Printf("  %s: %d samples → unchanged (cached)\n", name, len(files))
        continue
    }

    var embeddings []types.SampleEmbedding
    for _, file := range files {
        // Convert + Read audio
        audio.ConvertToWAV(bins.FFmpeg, file, tempWav)
        samples, sampleRate, _ := audio.ReadWAV(tempWav)

        // Extract embedding
        emb, _ := extractor.Extract(samples, sampleRate)

        embeddings = append(embeddings, types.SampleEmbedding{
            File:      filepath.Base(file),
            Hash:      hash,
            Embedding: emb,
        })
    }

    profile := types.SpeakerProfile{
        Name:       name,
        Embeddings: embeddings,
        Dim:        extractor.Dim(),
        Model:      "campplus_sv_zh-cn",
        ...
    }
    store.SaveProfile(profile)
}
```

### Metadata 更新

transcribe 完成後更新 Metadata：
```go
Metadata{
    SpeakersDetected:   countUniqueSpeakers(segments),
    SpeakersIdentified: countIdentifiedSpeakers(segments),  // 非 "Unknown" 的數量
}
```

---

## 7. 新增/修改檔案清單

### 新增

| 檔案 | 用途 |
|------|------|
| `internal/speaker/extractor.go` | CAM++ embedding extraction via sherpa-onnx |
| `internal/speaker/extractor_test.go` | Extractor 測試 |
| `internal/emotion/classifier.go` | SenseVoice emotion classification |
| `internal/emotion/classifier_test.go` | Classifier 測試 |

### 修改

| 檔案 | 修改內容 |
|------|---------|
| `go.mod` | 新增 sherpa-onnx-go-macos 依賴 |
| `internal/embedded/embedded.go` | BinPaths 移除 ONNXRuntime 欄位 |
| `internal/embedded/extract.go` | 移除 libonnxruntime binSpec + data var |
| `internal/embedded/embed_prod.go` | 移除 onnxruntime embed 宣告 |
| `internal/embedded/embedded_test.go` | 更新測試 |
| `internal/models/registry.go` | 新增 campplus + sensevoice 模型 |
| `internal/models/manager.go` | 支援 .onnx 檔名 |
| `cmd/commands/transcribe.go` | 加入 speaker/emotion 推理 |
| `cmd/commands/enroll.go` | 加入真實 embedding 計算 |
| `scripts/prepare-all.sh` | 移除 onnxruntime 驗證 |

---

## 8. 驗證計畫

### 單元測試

| 模組 | 測試內容 |
|------|---------|
| `speaker/extractor` | NewExtractor 建立/關閉、Extract 回傳正確維度 |
| `emotion/classifier` | NewClassifier 建立/關閉、Classify 回傳合法 emotion label |
| `models/registry` | 新增模型存在、欄位正確 |
| `embedded` | 調整後的 ExtractAll 只提取 2 個 binary |

### 整合測試

1. **Enroll E2E**：
   - 建立 `speakers/TestUser/` 放入測試 WAV
   - 執行 `enroll` → 確認 `.profile.json` 產生且包含 512 維 embedding
   - 再次 `enroll` → 確認 "unchanged"（cache hit）
   - 新增一個 WAV → `enroll` → 確認 "updated"

2. **Transcribe E2E**：
   - 準備已 enroll 的 speaker + 測試音檔
   - `transcribe --input test.wav` → 驗證 Speaker 欄位非 "Unknown"
   - 驗證 Emotion 欄位有真實標籤
   - 驗證 AudioEvent 欄位

3. **Model Auto-Download**：
   - 清空 models 目錄 → transcribe/enroll → 自動下載 CAM++ + SenseVoice

### 手動驗證

- 用 2-3 分鐘多人對話音檔測試 speaker identification 準確度
- 驗證不同情緒段落的 emotion label 合理性
