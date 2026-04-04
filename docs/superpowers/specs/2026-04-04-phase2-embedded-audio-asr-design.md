# Phase 2 設計規格：Embedded Binaries + Audio 轉換 + ASR 子程序

> Phase 2 of meeting-emo-transcriber — 建立二進位打包基礎設施、音訊處理管線、whisper-cli ASR 整合。

## 1. 背景與目標

### 為什麼做

Phase 1 完成了 Go 骨架（types、config、speaker store/matcher、output formatters、CLI skeleton）。
但 `transcribe` 命令仍是 stub — 無法實際處理音訊。

Phase 2 的目標是讓 `transcribe` 命令能**端到端執行 ASR**：
輸入任意格式音檔 → 輸出時間戳逐字稿（TXT/JSON/SRT）。

Speaker identification 和 emotion recognition 留給 Phase 3（ONNX 模型推理）。

### 預期成果

Phase 2 完成後：
- `scripts/prepare-all.sh` 一鍵準備所有外部 binary
- `go build -tags embed` 編譯出包含三個嵌入 binary 的單一執行檔
- `meeting-emo-transcriber transcribe --input meeting.mp3` 可輸出逐字稿
- 模型首次使用時自動下載

### 範圍

| 包含 | 不包含 |
|------|--------|
| go:embed 三個 binary（whisper-cli、ffmpeg、libonnxruntime.dylib） | ONNX 模型推理（CAM++、SenseVoice） |
| 自動化建置/下載腳本 | Speaker identification |
| ffmpeg 音訊格式轉換 | Emotion recognition |
| whisper-cli ASR + SRT 解析 | Unknown speaker auto-discovery |
| 模型下載管理（Whisper GGML + Silero VAD） | Pipeline 整合（Phase 4） |
| transcribe 命令實作（ASR-only 模式） | |

---

## 2. Embedded Binary 套件

### 2.1 目錄結構

```
internal/embedded/
├── embedded.go          # go:embed 宣告 + ExtractAll() 主函數
└── hash.go              # SHA-256 hash 比對（快取驗證）

embedded/binaries/
└── darwin-arm64/        # git-ignored，由腳本產生
    ├── whisper-cli
    ├── ffmpeg
    └── libonnxruntime.dylib

scripts/
├── build-whisper.sh         # 編譯 whisper-cli（Metal GPU）
├── download-ffmpeg.sh       # 下載 ffmpeg static binary
├── download-onnxruntime.sh  # 下載 libonnxruntime.dylib
└── prepare-all.sh           # 一鍵執行全部
```

### 2.2 嵌入的 Binary

| Binary | 來源 | 版本 | 用途 | 大小估計 |
|--------|------|------|------|----------|
| `whisper-cli` | whisper.cpp 編譯 | v1.7.3 | ASR + VAD | ~5MB |
| `ffmpeg` | 預編譯 static binary | 7.1 (pin) | 音訊格式轉換 | ~30MB |
| `libonnxruntime.dylib` | GitHub releases | v1.19.0 | ONNX 推理引擎 | ~20MB |

### 2.3 建置腳本

**`scripts/build-whisper.sh`**：
1. Clone whisper.cpp（或使用已存在的目錄）
2. Checkout tag `v1.7.3`
3. `cmake -B build -DGGML_METAL=1 -DGGML_METAL_EMBED_LIBRARY=1`
4. `cmake --build build --config Release -j$(sysctl -n hw.ncpu)`
5. 複製 `build/bin/whisper-cli` 到 `embedded/binaries/darwin-arm64/whisper-cli`
6. 下載 Silero VAD model → `embedded/binaries/darwin-arm64/ggml-silero-v6.2.0.bin`（或放到 models 目錄）

**`scripts/download-ffmpeg.sh`**：
1. 下載 macOS arm64 static ffmpeg binary
2. 驗證 SHA-256 checksum
3. 複製到 `embedded/binaries/darwin-arm64/ffmpeg`

**`scripts/download-onnxruntime.sh`**：
1. 下載 `onnxruntime-osx-arm64-1.19.0.tgz` from GitHub releases
2. 解壓取出 `lib/libonnxruntime.1.19.0.dylib`
3. 重命名為 `libonnxruntime.dylib`
4. 複製到 `embedded/binaries/darwin-arm64/`

**`scripts/prepare-all.sh`**：順序執行以上三個 + 檢查所有 binary 存在並列印 summary。

### 2.4 快取目錄

```
~/.meeting-emo-transcriber/
├── bin/
│   ├── whisper-cli              # 0755
│   ├── ffmpeg                   # 0755
│   ├── libonnxruntime.dylib     # 0644
│   └── .versions.json           # 已解壓 binary 的 SHA-256 記錄
└── models/
    └── (見 §4)
```

### 2.5 快取邏輯

1. 計算嵌入 binary 的 SHA-256 hash（編譯時固定）
2. 讀取 `~/.meeting-emo-transcriber/bin/.versions.json`
3. 逐一比對 hash：
   - 不同 → 寫出檔案、設定權限、更新 `.versions.json`
   - 相同 → 跳過（快取命中）
4. 回傳 `BinPaths` 結構

### 2.6 API

```go
package embedded

// BinPaths 所有解壓後的 binary 路徑
type BinPaths struct {
    WhisperCLI    string // ~/.meeting-emo-transcriber/bin/whisper-cli
    FFmpeg        string // ~/.meeting-emo-transcriber/bin/ffmpeg
    ONNXRuntime   string // ~/.meeting-emo-transcriber/bin/libonnxruntime.dylib
}

// ExtractAll 解壓所有嵌入的 binary 到快取目錄
// 使用 SHA-256 hash 判斷是否需要重新解壓
func ExtractAll() (BinPaths, error)

// CacheDir 回傳快取根目錄路徑
func CacheDir() string
```

---

## 3. Audio 模組

### 3.1 目錄結構

```
internal/audio/
├── convert.go     # ffmpeg 格式轉換
├── detect.go      # ffmpeg -i 格式偵測（不需 ffprobe）
├── wav.go         # WAV 讀取 + 片段切割
└── audio_test.go
```

### 3.2 格式轉換

**ffmpeg 命令**：
```bash
ffmpeg -y -i <input> -acodec pcm_s16le -ar 16000 -ac 1 <output.wav>
```

**參數說明**：
| Flag | 值 | 用途 |
|------|----|------|
| `-y` | — | 覆蓋已存在檔案 |
| `-i` | input path | 輸入音訊檔 |
| `-acodec` | `pcm_s16le` | 16-bit signed PCM little-endian |
| `-ar` | `16000` | 16kHz sample rate（whisper 要求） |
| `-ac` | `1` | mono channel |

**支援輸入格式**：`.wav, .mp3, .m4a, .flac, .ogg, .opus, .aac, .mp4, .mkv, .webm`

### 3.3 格式偵測（最佳化）

用 `ffmpeg -i` 的 stderr 輸出偵測輸入音訊格式（不需額外 embed ffprobe）：
```bash
ffmpeg -i <input> -f null - 2>&1
```

從 stderr 解析 stream 資訊（codec、sample rate、channels）。
若已是 `pcm_s16le, 16000 Hz, mono` → 跳過轉換。

> 選擇用 `ffmpeg -i` 而非 ffprobe，避免額外 embed 一個 binary。

### 3.4 WAV 讀取 + 片段切割

```go
// ReadWAV 讀取 WAV 檔案為 float32 samples
// 回傳 samples, sample rate, error
func ReadWAV(path string) ([]float32, int, error)

// ExtractSegment 依時間範圍切割音訊片段
// start/end 單位為秒
func ExtractSegment(samples []float32, sampleRate int, start, end float64) []float32

// WriteWAV 將 float32 samples 寫入 WAV 檔案
// 用於 unknown speaker auto-discovery 時儲存片段（Phase 4）
func WriteWAV(path string, samples []float32, sampleRate int) error
```

### 3.5 暫存檔策略

轉換產出的 WAV 放在系統 temp 目錄，transcribe 完成後清理：
```go
tempDir, _ := os.MkdirTemp("", "met-transcribe-*")
defer os.RemoveAll(tempDir)
wavPath := filepath.Join(tempDir, "input.wav")
```

---

## 4. 模型管理

### 4.1 目錄結構

```
internal/models/
├── manager.go       # 下載/驗證/路徑管理
├── registry.go      # 模型清單（URL、SHA-256）
└── models_test.go
```

### 4.2 模型清單（Phase 2 範圍）

| 模型 | 檔名 | 大小 | 來源 | 用途 |
|------|------|------|------|------|
| Whisper Large-v3 | `ggml-large-v3.bin` | ~3.1GB | HuggingFace | ASR（auto/en 預設） |
| Silero VAD v6.2.0 | `ggml-silero-v6.2.0.bin` | ~2MB | whisper.cpp repo | VAD |

Phase 3 時新增：
- `campplus_sv_zh-cn.onnx`（~30MB）— Speaker embedding
- `sensevoice-small-int8.onnx`（~66MB）— Emotion classification

### 4.3 ASR 模型自動選擇

| `--language` 值 | 選擇的模型 | 模型檔名 |
|-----------------|-----------|----------|
| `auto` | Large-v3 | `ggml-large-v3.bin` |
| `en` | Large-v3 | `ggml-large-v3.bin` |
| `zh-TW` | Breeze | `ggml-breeze-*.bin` |
| `zh` | Belle | `ggml-belle-*.bin` |
| `ja` | Kotoba | `ggml-kotoba-*.bin` |

> 注意：Phase 2 先只支援 Large-v3（auto/en）。語言專用模型的 GGML 轉換和 registry 可在後續迭代補充。

### 4.4 快取目錄

```
~/.meeting-emo-transcriber/models/
├── ggml-large-v3.bin
├── ggml-silero-v6.2.0.bin
└── .manifest.json        # 已下載模型的 hash + 下載時間
```

### 4.5 API

```go
package models

type ModelInfo struct {
    Name     string // 模型識別名
    URL      string // 下載 URL
    SHA256   string // 預期 hash
    Size     int64  // 檔案大小（bytes，用於進度顯示）
    Category string // "asr" | "vad" | "speaker" | "emotion"
}

// EnsureModel 確保模型存在，不存在則下載
// 顯示下載進度（大檔案）
func EnsureModel(name string) (modelPath string, err error)

// ResolveASRModel 根據 language 選擇對應 ASR 模型名
func ResolveASRModel(language string) string
```

---

## 5. ASR 模組（whisper-cli wrapper）

### 5.1 目錄結構

```
internal/asr/
├── whisper.go        # whisper-cli 子程序管理 + 呼叫
├── parser.go         # SRT 輸出解析 → []ASRResult
└── asr_test.go
```

### 5.2 whisper-cli 參數

Phase 2 使用的完整參數列表：

| 參數 | 值 | 說明 |
|------|----|------|
| `-m` | model path | GGML 模型檔路徑 |
| `-f` | wav path | 16kHz mono WAV 輸入 |
| `-l` | language code | `auto` / `zh` / `en` / `ja` |
| `-t` | thread count | CPU 執行緒數（預設 config.Threads） |
| `-osrt` | — | 輸出 SRT 格式 |
| `-of` | output base path | 輸出檔案路徑（不含副檔名） |
| `--no-prints` | — | 只輸出結果 |
| `--vad` | — | 啟用 Silero VAD |
| `-vm` | vad model path | Silero VAD model 路徑 |

**完整參數清單（備查，Phase 2 不一定全用）**：

```
基本控制：
  -t N       threads (default: 4)
  -p N       processors (default: 1)
  -ot N      offset time ms (default: 0)
  -d N       duration ms (default: 0, 全部)
  -mc N      max context tokens (default: -1)
  -ml N      max segment length chars (default: 0)
  -sow       split on word (default: false)

解碼策略：
  -bo N      best-of candidates (default: 5)
  -bs N      beam size (default: 5)
  -tp N      temperature (default: 0.0)
  -tpi N     temperature increment (default: 0.2)
  -et N      entropy threshold (default: 2.4)
  -lpt N     logprob threshold (default: -1.0)
  -nth N     no-speech threshold (default: 0.6)

輸出格式：
  -otxt      TXT 輸出
  -osrt      SRT 輸出
  -ovtt      VTT 輸出
  -ocsv      CSV 輸出
  -oj        JSON 輸出
  -ojf       JSON full（含更多資訊）
  -olrc      LRC 輸出
  -of FNAME  輸出路徑（不含副檔名）

功能開關：
  -tr        翻譯為英文
  -di        stereo diarization
  -tdrz      tinydiarize
  -nf        不用 temperature fallback
  --np       不印 debug
  --vad      啟用 Silero VAD
  -vm PATH   VAD model 路徑
  -ng        停用 GPU
  -fa        Flash Attention
  -sns       suppress non-speech tokens
```

### 5.3 SRT Parser

**輸入格式**：
```
1
00:00:00,000 --> 00:00:03,500
Hello, this is a test.

2
00:00:03,500 --> 00:00:07,000
今天的會議開始了。
```

**解析邏輯**：
1. 以空行分割 blocks
2. 每個 block：跳過序號行 → 解析時間戳 → 取文字
3. 時間戳格式：`HH:MM:SS,mmm --> HH:MM:SS,mmm` → 轉為 float64 秒

**輸出**：`[]types.ASRResult{Start, End, Text, Language}`

> Language 欄位：若使用 `auto` 模式，whisper-cli 的 SRT 不包含語言標記。
> 暫時由傳入的 `--language` 參數填入。Phase 3 可用 SenseVoice 的語言偵測補充。

### 5.4 API

```go
package asr

type WhisperConfig struct {
    BinPath      string // whisper-cli binary 路徑
    ModelPath    string // GGML model 路徑
    VADModelPath string // Silero VAD model 路徑
    Language     string // "auto" | "zh-TW" | "zh" | "en" | "ja"
    Threads      int    // CPU threads
}

// Transcribe 執行 whisper-cli 子程序，回傳 ASR 結果
// 1. 建立 temp output 路徑
// 2. 組裝 whisper-cli 參數
// 3. exec.Command 執行
// 4. 讀取 .srt 檔案 → ParseSRT()
func Transcribe(cfg WhisperConfig, wavPath string) ([]types.ASRResult, error)

// ParseSRT 解析 SRT 格式字串為 ASRResult slice
func ParseSRT(content string) ([]types.ASRResult, error)

// ParseSRTTimestamp 解析 "HH:MM:SS,mmm" 為 float64 秒
func ParseSRTTimestamp(ts string) (float64, error)
```

---

## 6. CLI 整合：transcribe 命令

### 6.1 執行流程

```
meeting-emo-transcriber transcribe --input meeting.mp3
    │
    ├─ 1. embedded.ExtractAll()
    │      → 解壓 whisper-cli, ffmpeg, libonnxruntime.dylib
    │
    ├─ 2. models.EnsureModel(ResolveASRModel(language))
    │      → 確保 GGML model 存在（首次使用自動下載）
    │
    ├─ 3. models.EnsureModel("silero-vad-v6.2.0")
    │      → 確保 VAD model 存在
    │
    ├─ 4. audio.ConvertToWAV(ffmpegPath, inputPath, tempWavPath)
    │      → 轉換為 16kHz mono WAV（若已是正確格式則跳過）
    │
    ├─ 5. asr.Transcribe(whisperCfg, tempWavPath)
    │      → whisper-cli 子程序 ASR → []ASRResult
    │
    ├─ 6. 組裝 TranscriptResult
    │      → ASRResult → TranscriptSegment（Speaker="Unknown", Emotion=Neutral）
    │      → 填入 Metadata（file, duration, date）
    │
    ├─ 7. output.Format(result, format)
    │      → 用 Phase 1 的 TXT/JSON/SRT formatter 輸出
    │
    └─ 8. 清理暫存檔
```

### 6.2 Phase 2 的 TranscriptSegment 填充

```go
for _, asr := range asrResults {
    segments = append(segments, types.TranscriptSegment{
        Start:      asr.Start,
        End:        asr.End,
        Speaker:    "Unknown",
        Emotion:    types.EmotionInfo{Label: "Neutral", Display: ""},
        AudioEvent: "Speech",
        Language:   asr.Language,
        Text:       asr.Text,
        Confidence: types.Confidence{Speaker: 0, Emotion: 0},
    })
}
```

### 6.3 輸出範例（Phase 2）

**TXT**：
```
Unknown
Hello, this is a test.
今天的會議開始了。
讓我們看一下第一季的數據。
```

**SRT**：
```
1
00:00:00,000 --> 00:00:03,500
(Unknown) Hello, this is a test.

2
00:00:03,500 --> 00:00:07,000
(Unknown) 今天的會議開始了。
```

---

## 7. 新增/修改檔案清單

### 新增

| 檔案 | 用途 |
|------|------|
| `internal/embedded/embedded.go` | go:embed + ExtractAll |
| `internal/embedded/hash.go` | SHA-256 快取驗證 |
| `internal/audio/convert.go` | ffmpeg 格式轉換 |
| `internal/audio/detect.go` | ffprobe 格式偵測 |
| `internal/audio/wav.go` | WAV 讀取 + 片段切割 |
| `internal/audio/audio_test.go` | 音訊模組測試 |
| `internal/asr/whisper.go` | whisper-cli wrapper |
| `internal/asr/parser.go` | SRT 解析器 |
| `internal/asr/asr_test.go` | ASR 模組測試 |
| `internal/models/manager.go` | 模型下載管理 |
| `internal/models/registry.go` | 模型清單 |
| `internal/models/models_test.go` | 模型管理測試 |
| `embedded/binaries/.gitkeep` | 佔位（binary 被 gitignore） |
| `scripts/build-whisper.sh` | 編譯 whisper-cli |
| `scripts/download-ffmpeg.sh` | 下載 ffmpeg |
| `scripts/download-onnxruntime.sh` | 下載 libonnxruntime |
| `scripts/prepare-all.sh` | 一鍵準備 |

### 修改

| 檔案 | 修改內容 |
|------|---------|
| `cmd/commands/transcribe.go` | 從 stub 改為完整實作 |
| `cmd/commands/enroll.go` | 加入 audio.ConvertToWAV 驗證 |
| `.gitignore` | 加入 `embedded/binaries/darwin-arm64/` |
| `go.mod` | 不需新增依賴（已有 go-audio/wav） |

---

## 8. 驗證計畫

### 單元測試

| 模組 | 測試內容 |
|------|---------|
| `embedded` | ExtractAll 寫出檔案、權限正確、hash 快取命中/失效 |
| `audio` | WAV 讀取、格式偵測、ConvertToWAV（需真實 ffmpeg）|
| `asr/parser` | SRT 解析（多 block、各種時間戳、多行文字、空行處理）|
| `models` | ResolveASRModel 語言→模型對應 |

### 整合測試

1. **Binary Embedding E2E**：
   - `scripts/prepare-all.sh` 成功執行
   - `go build -tags embed` 編譯成功
   - 執行後 `~/.meeting-emo-transcriber/bin/` 出現三個 binary
   - 第二次執行快取命中（hash 不變）

2. **Audio Conversion E2E**：
   - 輸入 MP3 → 輸出 WAV 16kHz mono
   - 輸入已正確格式的 WAV → 跳過轉換
   - 不支援的格式 → 明確錯誤訊息

3. **ASR E2E**：
   - 準備短音檔（5-10 秒）
   - `transcribe --input test.wav --format txt` → 產出文字
   - `transcribe --input test.mp3 --format srt` → 產出 SRT
   - `transcribe --input test.wav --format all` → 產出三個檔案

4. **Model Auto-Download**：
   - 清空 models 目錄 → `transcribe` → 自動下載 + 進度顯示
   - 再次執行 → 跳過下載

### 手動驗證

- `otool -L` 確認編譯產物無意外動態依賴
- 實際用一段 2-3 分鐘的會議錄音測試 ASR 品質
