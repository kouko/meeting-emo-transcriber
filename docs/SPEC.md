# Meeting Emotion Transcriber — 實作規格書

> 開發用藍圖。閱讀此文件即可開始實作。

## 1. 專案概述

### 目標

開發一個 macOS CLI 工具（Go 語言），對會議錄音進行：

1. **語音轉文字（ASR）** — 高精度中文轉錄（支援中英日混雜）
2. **發話者辨識（Speaker Identification）** — 預註冊聲紋，標註「誰在說話」
3. **情感辨識（SER）** — 辨識每段發話的情緒狀態（預設最多 7 類+unk，可選 9 類）

最終編譯為單一 macOS Binary（Mach-O） + 模型檔案，使用者無需安裝 Python。

### Non-Goals

- GUI 介面
- HTTP Server / API
- 即時錄音 / 串流處理
- 模型微調 / 訓練
- 跨平台（僅 macOS arm64）

### 核心約束

| 約束 | 說明 |
|------|------|
| 語言 | Go 1.25+（主 CLI，允許 cgo）+ Swift 6.0（metr-diarize） |
| 部署 | 單一 Binary（embed.FS 嵌入所有依賴，無 build tag）+ 模型檔（首次執行自動下載） |
| 處理模式 | 離線批次 |
| 隱私 | 全程本地運算，不上傳雲端 |

---

## 2. 技術架構

### 方案：元件組合式（Component-based）+ embed.FS 單一 Binary

追求各模組最高精度。採用 `embed.FS`（Go 標準庫，無需 build tag）將所有外部依賴嵌入 Go Binary，使用者只需要一個執行檔 + 模型檔案。

### 打包策略

- **whisper-cli**：編譯為獨立 CLI Binary → `embed.FS` 嵌入 Go Binary → 執行時解壓到 `~/.metr/bin/` → 透過 `os/exec` 子程序呼叫
- **ffmpeg**：同上，用於音訊格式轉換
- **metr-diarize**：Swift 6.0 CLI（FluidAudio CoreML/ANE）→ `embed.FS` 嵌入 → 執行時解壓到 `~/.metr/bin/` → 透過 `os/exec` 子程序呼叫
- **效果**：使用者拿到的是**單一 Go Binary**，首次執行時自動解壓內嵌的依賴到 `~/.metr/bin/`

```
┌──────────────────────────────────────────────────────────────┐
│                     metr (Go Binary, Mach-O)                  │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  embed.FS（嵌入的外部依賴，無 build tag）               │     │
│  │                                                        │     │
│  │  whisper-cli    ffmpeg    metr-diarize                  │     │
│  │  (ASR+VAD)     (音訊轉換)  (FluidAudio Swift CLI)       │     │
│  └───────┬─────────────┬─────────────┬─────────────────── ┘     │
│          │             │             │ 執行時解壓到 ~/.metr/bin/  │
│          ▼             ▼             ▼                           │
│  ┌───────────┐  ┌──────────┐  ┌────────────────────────┐       │
│  │ os/exec   │  │ os/exec  │  │      os/exec           │       │
│  │whisper-cli│  │  ffmpeg  │  │   metr-diarize         │       │
│  │ ASR + VAD │  │  WAV 轉換 │  │ FluidAudio CoreML/ANE  │       │
│  │(Metal GPU)│  │          │  │ 說話者分離              │       │
│  └───────────┘  └──────────┘  └────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  sherpa-onnx-go-macos (cgo, in-process)               │       │
│  │  CAM++ 192d (Speaker Embedding + Matching)             │       │
│  │  SenseVoice-Small int8 (Emotion + Audio Event)         │       │
│  └──────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 快取目錄結構

```
~/.metr/
├── bin/
│   ├── whisper-cli              # whisper.cpp CLI Binary
│   ├── ffmpeg                   # 音訊格式轉換
│   └── metr-diarize             # FluidAudio Swift CLI
└── models/
    └── diarization/             # FluidAudio 使用的 CoreML 模型（自動下載）
```

首次執行時解壓嵌入的 Binary 到快取目錄。後續啟動時檢查檔案 hash，若一致則直接使用快取。

### 技術棧

| 元件 | 選型 | 角色 | 整合方式 |
|------|------|------|---------|
| ASR（預設） | whisper.cpp CLI + Whisper Large-v3 (GGML) | 語音轉文字 + Metal GPU 加速 | `embed.FS` + `os/exec` 子程序 |
| ASR（台灣中文推薦） | whisper.cpp CLI + Breeze ASR 25 (GGML) | 台灣口音 + 繁體直出 + 中英混合 | 同上，僅切換模型檔案 |
| 說話者分離 | FluidAudio + metr-diarize (Swift 6.0 CLI) | CoreML/ANE 加速說話者分離 | `embed.FS` + `os/exec` 子程序 |
| Speaker Embedding | CAM++ zh-cn (ONNX, 192d) | 聲紋向量提取 + 身份比對 | `sherpa-onnx-go-macos`（cgo, in-process） |
| SER（預設） | SenseVoice-Small (ONNX int8, 最多 7 類+unk) | 情感辨識 | `sherpa-onnx-go-macos`（cgo, in-process，~228MB tar.bz2） |
| VAD | silero-vad v6.2.0 | 語音活動偵測 | whisper CLI 參數控制 |
| 音訊轉換 | ffmpeg CLI | MP3/M4A/FLAC → WAV 16kHz mono | `embed.FS` + `os/exec` 子程序 |
| 音訊 I/O | `go-audio/wav` | 讀取 WAV PCM | pure Go |
| 向量運算 | `gonum` | Cosine similarity | pure Go |
| Config | `spf13/viper` | YAML 設定檔 | pure Go |
| CLI | `spf13/cobra` | 命令列介面 | pure Go |
| 嵌入 | `embed.FS` (標準庫) | 嵌入 3 個 CLI Binary（無 build tag） | pure Go |

---

## 3. 目錄結構

```
meeting-emo-transcriber/
├── cmd/
│   └── main.go                  # CLI 入口
├── embedded/
│   ├── embedded.go              # embed.FS 宣告 + 執行時解壓邏輯（無 build tag）
│   └── binaries/                # 預編譯的平台 Binary（.gitignore, build 時產生）
│       └── darwin-arm64/
│           ├── whisper-cli      # whisper.cpp CLI Binary
│           ├── ffmpeg           # 音訊格式轉換
│           └── metr-diarize     # FluidAudio Swift CLI
├── tools/
│   └── metr-diarize/            # Swift 6.0 CLI 原始碼（FluidAudio 說話者分離）
├── internal/
│   ├── config/
│   │   └── config.go            # YAML 設定載入
│   ├── audio/
│   │   └── audio.go             # 音訊讀取、重採樣、片段切割（支援 WAV/MP3/M4A）
│   ├── vad/
│   │   └── vad.go               # VAD 語音切割（whisper.cpp 內建 VAD 參數）
│   ├── asr/
│   │   └── asr.go               # whisper.cpp 子程序呼叫封裝 + SRT 快取
│   ├── diarize/
│   │   └── diarize.go           # metr-diarize 子程序呼叫封裝（FluidAudio）
│   ├── speaker/
│   │   ├── extractor.go         # CAM++ Embedding 提取（onnxruntime-go）
│   │   ├── matcher.go           # Cosine similarity 比對
│   │   └── store.go             # 資料夾驅動聲紋管理 + .profile.json 快取
│   ├── emotion/
│   │   └── emotion.go           # SenseVoice / Emotion2vec+ 情感辨識（onnxruntime-go）
│   ├── pipeline/
│   │   └── pipeline.go          # 主處理管線：串接所有模組
│   └── output/
│       ├── json.go              # JSON 輸出
│       ├── txt.go               # TXT 純文字逐字稿輸出
│       └── srt.go               # SRT 字幕輸出
├── scripts/
│   └── build-whisper.sh         # whisper.cpp 編譯腳本（平台偵測 + Metal 支援）
├── models/                      # 模型檔案（.gitignore）
├── docs/
│   └── SPEC.md                  # 本文件
├── Makefile                     # 統一編譯入口
├── go.mod
└── go.sum
```

### 使用者端的工作目錄（非 repo 內容）

```
<任意工作目錄>/
├── speakers/                    # 聲紋資料庫（可攜式，整個資料夾複製即用）
│   ├── config.yaml              #   選用：這個資料夾專用的設定
│   ├── CEO_Wang/                #   子資料夾名 = 講者名
│   │   ├── office.wav           #   音訊樣本（支援 wav/mp3/m4a/flac/ogg）
│   │   ├── home.m4a
│   │   └── .profile.json        #   自動產生的 embedding 快取
│   └── Manager_Lin/
│       ├── mic.wav
│       └── .profile.json
├── output/                      # 轉錄結果（自動建立）
│   ├── meeting-2026-04-04.txt
│   └── standup-2026-04-01.json
└── meeting.wav                  # 待轉錄的音檔
```

### 雙層架構：全局 vs 可攜

| 層級 | 位置 | 內容 | 可攜？ |
|------|------|------|--------|
| **全局** | `~/.metr/` | `bin/`（whisper-cli + ffmpeg + metr-diarize）+ `models/`（ASR + Speaker + Emotion + Diarization） | 否，安裝一次共用 |
| **可攜** | `./speakers/` | 聲紋 + config.yaml + .profile.json | **是**，整個複製即用 |
| **輸出** | `./output/` | 轉錄結果 | 使用者自行管理 |

### speakers/ 資料夾說明

- **可攜式設計**：整個 `speakers/` 資料夾複製到另一台電腦即可直接使用（只要該電腦有安裝本工具）
- **資料夾名即講者名**：子資料夾名稱直接作為講者的辨識名稱
- **多格式支援**：支援 `.wav`、`.mp3`、`.m4a`、`.flac`、`.ogg` 等常見音訊格式
- **自動快取**：`.profile.json` 由程式自動產生與維護，使用者不需手動編輯
- **內含設定**：`speakers/config.yaml`（選用）跟著資料夾走，不同 speakers 資料夾可有不同設定

---

## 4. 資料結構

```go
// --- 核心型別 ---

// AudioSegment 代表從原始音訊中依 ASRResult 時間戳切出的片段
// 用於 Speaker Embedding 提取和 Emotion 辨識
type AudioSegment struct {
    Start    float64   // 起始秒數
    End      float64   // 結束秒數
    Audio    []float32 // 16kHz mono PCM samples
}

// TranscriptSegment 最終輸出的單一段落
type TranscriptSegment struct {
    Start      float64  `json:"start"`
    End        float64  `json:"end"`
    Speaker    string        `json:"speaker"`
    Emotion    EmotionInfo  `json:"emotion"`
    AudioEvent string       `json:"audio_event"` // SenseVoice 音訊事件：Speech, Laughter, Applause 等
    Language   string       `json:"language"`
    Text       string   `json:"text"`
    Confidence struct {
        Speaker float32 `json:"speaker"`
        Emotion float32 `json:"emotion"`
    } `json:"confidence"`
}

// TranscriptResult 完整轉錄結果
type TranscriptResult struct {
    Metadata struct {
        File             string `json:"file"`
        Duration         string `json:"duration"`
        SpeakersDetected int    `json:"speakers_detected"`
        SpeakersIdentified int  `json:"speakers_identified"`
        Date             string `json:"date"`
    } `json:"metadata"`
    Segments []TranscriptSegment `json:"segments"`
}

// --- Speaker 相關 ---

// SpeakerProfile 已註冊的講者聲紋（對應 .profile.json）
type SpeakerProfile struct {
    Name      string          `json:"name"`
    Embeddings []SampleEmbedding `json:"embeddings"` // 每個樣本獨立存 embedding
    Dim        int                `json:"dim"`        // 向量維度（CAM++ = 512）
    Model      string             `json:"model"`      // 使用的聲紋模型名稱
    CreatedAt  string             `json:"created_at"`
    UpdatedAt  string             `json:"updated_at"`
}

// SampleEmbedding 單一音訊樣本的 embedding + 檔案記錄
type SampleEmbedding struct {
    File      string    `json:"file"`      // 檔案名稱（相對於講者資料夾）
    Hash      string    `json:"hash"`      // SHA-256 hash，格式 "sha256:<hex>"
    Embedding []float32 `json:"embedding"` // 該樣本的 512d 向量
}

// MatchResult Speaker 比對結果
type MatchResult struct {
    Name       string  // 講者姓名，Unknown 表示未識別
    Similarity float32 // Cosine similarity 分數
}

// --- Emotion 相關 ---

// EmotionInfo JSON 輸出用的情緒三層結構
type EmotionInfo struct {
    Raw     string `json:"raw"`     // 模型原始輸出：HAPPY
    Label   string `json:"label"`   // 標準化：Happy
    Display string `json:"display"` // CC 副詞：happily（Neutral 為空）
}

// EmotionResult 情感辨識結果（三層字串定義）
type EmotionResult struct {
    Raw        string  // 模型原始輸出：HAPPY, SAD, ANGRY, NEUTRAL
    Label      string  // 標準化內部表示：Happy, Sad, Angry, Neutral
    Display    string  // CC Manner Caption 副詞：happily, sadly, angrily, ""（Neutral 為空）
    Confidence float32
}
// SenseVoice: Happy, Sad, Angry, Neutral, Fearful, Disgusted, Surprised, unk
//   （model.py 映射 5 token，README 列 7 類，需實測確認）
// Emotion2vec+: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown（9 類）
// 完整對應表見 §9「情緒字串對應表」
```

---

## 5. 模組介面

```go
// --- ASR（subprocess-based）---

type Transcriber interface {
    // TranscribeFile 對音檔進行語音辨識（透過 whisper-cli 子程序）
    // audioPath: 輸入音檔路徑（WAV 16kHz mono）
    // 回傳帶時間戳的文字片段（解析 whisper 的 SRT/JSON 輸出）
    TranscribeFile(audioPath string) ([]ASRResult, error)
}

type ASRResult struct {
    Start    float64
    End      float64
    Text     string
    Language string
}

// 實作說明：
//   1. 呼叫 whisper-cli 子程序：
//      exec.CommandContext(ctx, whisperPath, "-m", modelPath, "-f", audioPath, "-osrt", ...)
//   2. 解析 whisper 輸出的 SRT 檔案取得帶時間戳的文字
//   3. whisper-cli 路徑由 embedded 模組提供（從快取目錄取得）

// --- Speaker ---

type EmbeddingExtractor interface {
    // Extract 從音訊提取 speaker embedding 向量
    Extract(audio []float32, sampleRate int) ([]float32, error)
    // Dim 回傳向量維度（CAM++ = 512）
    Dim() int
    Close()
}

// MatchStrategy 聲紋比對策略介面（Strategy Pattern）
// 預留模組化空間，未來可替換演算法
type MatchStrategy interface {
    // Score 計算 segment embedding 與單一 profile 的相似度分數
    Score(segmentEmb []float32, profile SpeakerProfile) float32
}

// MaxSimilarityStrategy 預設策略：逐一比對所有樣本，取最高分
// 對不同錄音環境的樣本最強健
type MaxSimilarityStrategy struct{}

// CentroidStrategy 備選策略：先將多樣本 embedding 平均為 centroid，再比對
// 運算最快但環境差異大時精度下降
type CentroidStrategy struct{}

type SpeakerMatcher interface {
    // Match 將 embedding 與已註冊聲紋比對（使用注入的 MatchStrategy）
    // 若全部低於 threshold，回傳 MatchResult{Name: "", Similarity: bestScore}
    Match(embedding []float32, profiles []SpeakerProfile, threshold float32) MatchResult
}

// UnknownSpeakerTracker 管理轉錄過程中自動發現的未知講者
type UnknownSpeakerTracker interface {
    // TryMatch 嘗試與已發現的 unknown speakers 比對
    // 若匹配 → 回傳該 unknown 的名稱（如 "speaker_1"）
    // 若不匹配 → 建立新的 unknown speaker，回傳新名稱
    TryMatch(embedding []float32, audioSegment []float32, timestamp float64, threshold float32) (name string, err error)
    // Flush 轉錄結束後，將所有自動發現的講者寫入 speakers/ 資料夾
    Flush() error
}

type SpeakerStore interface {
    // ScanAndEnroll 掃描 speakers/ 資料夾，自動註冊/更新所有講者
    // force=true 時忽略快取，強制重新計算所有 embedding
    ScanAndEnroll(force bool) ([]EnrollResult, error)
    // LoadProfiles 載入所有已註冊聲紋（從 .profile.json 快取）
    LoadProfiles() ([]SpeakerProfile, error)
    // List 列出所有講者名稱
    List() ([]string, error)
    // Verify 用測試音檔驗證指定講者的辨識準確度
    Verify(name string, audioPath string) (MatchResult, error)
}

// EnrollResult 單一講者的 enroll 結果
type EnrollResult struct {
    Name    string
    Samples int    // 樣本數量
    Status  string // "created" | "updated" | "unchanged"
}

// 快取失效邏輯（SpeakerStore 內部實作）：
//   1. 掃描 speakers/<name>/ 下所有音訊檔（依副檔名過濾）
//   2. 計算每個檔案的 SHA-256 hash
//   3. 比對 .profile.json 中記錄的 hash
//   4. 若有差異（新增/修改/刪除）→ 重新提取所有樣本 embedding
//   5. 若完全一致 → 跳過，回傳 "unchanged"

// --- Emotion ---

type EmotionClassifier interface {
    // Classify 對音訊片段進行情感辨識
    Classify(audio []float32, sampleRate int) (EmotionResult, error)
    Close()
}

// --- VAD ---
// whisper.cpp 內建 VAD，透過 CLI 參數控制（如 --no-speech-thold）
// 不需要獨立 VAD interface，由 Transcriber 內部處理

// --- Output ---

type Formatter interface {
    // Format 將轉錄結果格式化為字串
    Format(result TranscriptResult) (string, error)
}

// --- Pipeline ---

type Pipeline interface {
    // Process 完整處理流程：Auto-Enroll → ASR(含VAD) → Speaker → Emotion → Merge
    Process(audioPath string) (TranscriptResult, error)
    Close()
}
```

---

## 6. 處理管線

### 8 階段管線（雙路平行）

```
會議音檔 (WAV/MP3/M4A/...)
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │ [1/8] 解壓嵌入 Binary                          │
  │  embedded.ExtractAll()                         │
  │  → ~/.metr/bin/{whisper-cli, ffmpeg, metr-diarize} │
  └──────────────────┬───────────────────────────┘
                     │
  ┌──────────────────▼───────────────────────────┐
  │ [2/8] 確保 ASR 模型（依語言自動選擇）            │
  │  models.EnsureASR()                            │
  │  zh-TW→Breeze / zh→Belle / ja→Kotoba / *→Large-v3 │
  └──────────────────┬───────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │ [3/8] 確保 VAD 模型                           │
  │  models.EnsureVAD()  → silero-vad-v6.2.0     │
  └──────────────────┬──────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────┐
  │ [4/8] 音訊轉換                                │
  │  audio.ConvertToWAV()                         │
  │  ffmpeg → 16kHz mono WAV（暫存檔）             │
  └──────────────────┬──────────────────────────┘
                     │ tempAudio.wav
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
  ┌────────────────┐  ┌───────────────────────────┐
  │ [5/8] ASR      │  │ [6/8] 說話者分離（平行）      │
  │ whisper-cli    │  │ metr-diarize subprocess   │
  │ VAD + ASR      │  │ FluidAudio CoreML/ANE     │
  │ → SRT（有快取）  │  │ → []DiarSegment           │
  │   若 SRT 快取   │  │   {start, end, speaker_id}│
  │   命中則跳過    │  └───────────────────────────┘
  └────────┬───────┘          │
           │ []ASRResult       │ []DiarSegment
           └──────────┬────────┘
                      │ 時間重疊合併
                      ▼
  ┌───────────────────────────────────────────┐
  │ [7/8] 講者身份解析                          │
  │  speaker.Resolve()                         │
  │  CAM++ 192d embedding（sherpa-onnx-go）    │
  │  vs. enrolled profiles（cosine similarity）│
  │  + Auto-Enroll（若 profiles 有變更）        │
  │  → speaker name per segment                │
  └───────────────────┬───────────────────────┘
                      │
  ┌───────────────────▼───────────────────────┐
  │ [8/8] 輸出                                 │
  │  output.Format()                           │
  │  TXT / JSON / SRT                          │
  └───────────────────────────────────────────┘
```

### 處理步驟

1. **[1/8] 解壓嵌入 Binary**：首次執行時將 whisper-cli、ffmpeg、metr-diarize 解壓到 `~/.metr/bin/`；後續啟動比對 hash，一致則直接使用快取
2. **[2/8] 確保 ASR 模型**：依 `--language` 自動選擇並下載對應 GGML 模型（首次執行時）
3. **[3/8] 確保 VAD 模型**：確保 silero-vad-v6.2.0 已下載
4. **[4/8] 音訊轉換**：ffmpeg 子程序將原始音檔轉為 16kHz mono WAV 暫存檔
5. **[5/8] 語音辨識（含 SRT 快取）**：呼叫 whisper-cli → 輸出 SRT → 解析為 `[]ASRResult`。若相同音檔的 SRT 快取存在則跳過 Whisper，直接讀取快取
6. **[6/8] 說話者分離（平行）**：呼叫 metr-diarize 子程序（FluidAudio CoreML/ANE 加速）→ 輸出 `[]DiarSegment{start, end, speaker_id}`
7. **[7/8] 講者身份解析**：將 ASRResult 與 DiarSegment 依時間重疊合併；Auto-Enroll（若 profiles 有變更則先重新計算 embedding）；CAM++ 192d 向量比對 enrolled profiles，識別真實講者姓名
8. **[8/8] 輸出**：依指定格式輸出 TXT / JSON / SRT

### 資料流跨程序邊界

```
Go 主程序                          whisper-cli 子程序
─────────                          ────────────────
寫入暫存 WAV ─── 檔案 ──→          讀取 WAV
                                   ASR + VAD 推理
讀取 SRT    ←── 檔案 ───           輸出 SRT
解析時間戳
切割音訊片段
Speaker/Emotion（in-process via ONNX Runtime）
```

> **設計取捨**：whisper.cpp 透過檔案 I/O 與主程序溝通（非記憶體共享），會有少量磁碟讀寫開銷。但避免了 cgo binding 的編譯複雜度，且可直接復用 `youtube-summarize-scraper` 的 build script。

> **已知風險：長時間會議**：超過 1 小時的會議錄音可能導致記憶體壓力（多個 ONNX 模型同時載入 + 大量音訊片段）。目前不做分段處理，實測後若遇到問題再加入「每 N 分鐘一段」的分段策略。

### Cosine Similarity 實作

```go
func CosineSimilarity(a, b []float32) float32 {
    var dot, normA, normB float32
    for i := range a {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
```

### 多樣本比對策略

預設採用 **Max Similarity**：不合併樣本，每次比對時逐一與所有樣本計算相似度，取最高分。

```
CEO_Wang 有 3 份樣本 (A: 辦公室, B: 家裡, C: 戶外)
輸入片段 X 的比對過程：

  sim(X, A) = 0.82
  sim(X, B) = 0.65    → max = 0.82 ≥ threshold → ✓ 匹配
  sim(X, C) = 0.71

vs Centroid（平均法）：
  sim(X, avg(A,B,C)) = 0.73  → 可能剛好低於 threshold → ✗ 誤判
```

**Max Similarity 的優勢**：
- 不同環境的錄音（辦公室、電話、戶外）不會互相「稀釋」
- 只要有一份樣本與輸入接近就能匹配
- 離線批次處理下，多次比對的效能開銷可忽略

**架構預留**：透過 `MatchStrategy` 介面（§5），未來可替換為 Centroid 或 Multi-centroid 等策略。

---

## 7. CLI 設計

```bash
# 最簡用法：當前目錄有 speakers/ 就自動偵測
metr transcribe --input meeting.wav
# → 自動偵測 ./speakers/，輸出到 ./output/meeting.txt

# 指定 speakers 資料夾
metr transcribe --input meeting.wav --speakers /path/to/my-speakers/

# 指定輸出格式
metr transcribe --input meeting.wav --format json

# 多格式同時輸出
metr transcribe --input meeting.wav --format txt,json,srt

# 全格式輸出
metr transcribe --input meeting.wav --format all
# → ./output/meeting.txt + ./output/meeting.json + ./output/meeting.srt

# 多格式 + --output：自動替換副檔名
metr transcribe --input meeting.wav --format txt,json --output ~/Desktop/result.json
# → ~/Desktop/result.txt + ~/Desktop/result.json

# 指定說話者數量（輔助 diarization）
metr transcribe --input meeting.wav --num-speakers 3

# 掃描 speakers/ 資料夾，預先註冊所有講者
metr enroll

# 強制重新計算（忽略快取）
metr enroll --force

# 列出已註冊聲紋
metr speakers list

# 驗證辨識準確度
metr speakers verify --name CEO_Wang --audio test.wav

# 初始化工作目錄（選用，建立 speakers/ 和 output/ 資料夾）
metr init
```

### 命令結構

```
metr
├── init            初始化工作目錄（建立 speakers/ + output/ + 範本 config.yaml）
├── enroll          掃描 speakers/ 資料夾，註冊所有講者
│   --force         強制重新計算所有 embedding（忽略快取）
├── transcribe      會議轉錄（自動 auto-enroll）
│   --input         音檔路徑（必填）
│   --output        輸出檔案路徑（選填，預設 ./output/<input-name>.<format>）
│   --format        輸出格式：txt | json | srt | all（預設 txt，可逗號分隔多值如 txt,json）
│   --language      語言：auto | zh-TW | zh | en | ja（預設 auto）
│   --threshold     Speaker 比對門檻（預設 0.7）
│   --num-speakers  預期說話者數量，0 表示自動偵測（預設 0）
├── speakers        聲紋管理
│   ├── list        列出所有已註冊講者（名稱、樣本數、最後更新時間）
│   └── verify      驗證辨識準確度
│       --name      講者名稱（必填）
│       --audio     測試音檔路徑（必填）
└── Global Flags
    --speakers      speakers 資料夾路徑（預設 ./speakers/）
    --config        設定檔路徑（選用，預設讀取 <speakers-dir>/config.yaml）
    --log-level     日誌等級：debug | info | warn | error
```

### enroll 輸出範例

```
$ metr enroll

Scanning speakers/...
  CEO_Wang:    3 samples → embedding computed ✓ (created)
  Manager_Lin: 2 samples → embedding computed ✓ (created)

2 speakers enrolled.

$ metr enroll

Scanning speakers/...
  CEO_Wang:    3 samples → unchanged (cached)
  Manager_Lin: 3 samples → embedding recomputed ✓ (updated, +1 sample)

1 speaker updated, 1 unchanged.
```

### speakers verify 輸出範例

```
$ metr speakers verify --name CEO_Wang --audio test_wang.wav

Verifying against CEO_Wang...
  Similarity: 0.89 (threshold: 0.70)
  Result: ✓ MATCH
```

### 講者管理方式

| 操作 | 方法 |
|------|------|
| **新增講者** | 建立 `speakers/<name>/` 資料夾，放入音訊樣本，執行 `enroll` |
| **新增樣本** | 將音訊檔放入對應的講者資料夾，執行 `enroll`（自動偵測變更）|
| **刪除講者** | 直接刪除 `speakers/<name>/` 資料夾 |
| **刪除樣本** | 直接刪除音訊檔，執行 `enroll`（自動偵測變更並重算 embedding）|
| **重新命名** | 重新命名資料夾即可（自動發現的 speaker_N 也適用） |
| **辨識新人** | 轉錄時自動發現，建立 `speaker_N/` 資料夾。使用者事後重新命名為真實姓名 |

### Exit Codes

| Code | 意義 |
|------|------|
| 0 | 成功 |
| 1 | 一般錯誤（設定錯誤、檔案不存在等）|
| 2 | 模型載入失敗 |
| 3 | 音訊格式不支援 |

---

## 8. 設定檔

### 設計原則：零設定即可執行

所有參數都有合理預設值，不需要 config.yaml 就能運行。config.yaml 是**選用的**，用於覆寫預設值。

### 設定來源優先順序

```
1. CLI flag（--threshold 0.7, --language zh, ...）     最高優先
2. --config 指定的檔案
3. <speakers-dir>/config.yaml                          跟著 speakers 資料夾走
4. 硬編碼預設值                                          最低優先
```

### config.yaml 範例（選用，放在 speakers/ 資料夾內）

```yaml
# speakers/config.yaml
# 只需要列出想覆寫的參數，其餘用預設值

language: "auto"          # auto | zh-TW | zh | en | ja
threshold: 0.7            # Speaker 比對門檻
format: txt               # 預設輸出格式
num_speakers: 0           # 預期說話者數量（0 = 自動偵測）
```

### 所有設定參數與預設值

| 參數 | CLI flag | config key | 預設值 | 說明 |
|------|----------|------------|--------|------|
| 語言 | `--language` | `language` | `"auto"` | ASR 語言（影響自動模型選擇，見下方） |
| 比對門檻 | `--threshold` | `threshold` | `0.7` | Speaker cosine similarity 門檻 |
| 輸出格式 | `--format` | `format` | `"txt"` | txt / json / srt / all（可逗號分隔多值） |
| 說話者數量 | `--num-speakers` | `num_speakers` | `0` | 預期說話者數量（0 = 自動偵測） |
| ASR 模型 | — | `models.whisper` | （依 language 自動選擇） | 明確指定時覆蓋自動選擇 |
| Speaker 模型 | — | `models.speaker` | `~/.metr/models/campplus_sv_zh-cn.onnx` | CAM++ ONNX 模型路徑 |
| Emotion 模型 | — | `models.emotion` | `~/.metr/models/sensevoice-small-int8/` | SER 模型路徑（預設 SenseVoice-Small int8） |
| 執行緒數 | — | `threads` | CPU 核心數 | ONNX Runtime + whisper 執行緒 |
| 比對策略 | — | `strategy` | `"max_similarity"` | max_similarity / centroid |
| 日誌等級 | `--log-level` | — | `"info"` | debug / info / warn / error |

### ASR 模型自動選擇邏輯

```
使用者指定了 models.whisper？
  ├─ 是 → 使用指定的模型（覆蓋自動選擇）
  └─ 否 → 依 language 設定自動選擇：
```

| language | 自動選擇的 ASR 模型 | 說明 |
|----------|-------------------|------|
| `zh-TW` | Breeze ASR 25 (GGML) | 台灣口音 + 繁體直出 + 中英混合 |
| `zh` | Belle-whisper-large-v3-turbo-zh (GGML) | 簡體中文特化 |
| `ja` | kotoba-whisper-v2.0 (GGML) | 日文特化 |
| `auto` / `en` / 其他 | Whisper Large-v3 (GGML) | 通用多語言 |

### 模型下載來源

所有模型首次使用時自動從 HuggingFace 下載到 `~/.metr/models/`：

```yaml
# 內建的模型下載 URL 對應（硬編碼）
model_sources:
  large-v3: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"
  breeze-asr-25: "https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp/resolve/main/ggml-model-q5_k.bin"
  belle-zh: "https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh-ggml/resolve/main/ggml-model.bin"
  kotoba-ja: "https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/main/ggml-model.bin"
```

> **注意**：模型路徑等技術參數通常不需要在 config.yaml 中設定。全局目錄 `~/.metr/models/` 是預設的模型位置，只有在非標準安裝時才需要覆寫。

---

## 9. 輸出格式

### 情緒字串對應表（三層定義）

情緒字串從 SER 模型輸出到使用者顯示經過三層轉換。

**SenseVoice 情緒標籤**（預設 SER 模型）：

來源：[SenseVoice model.py `emo_dict`](https://github.com/FunAudioLLM/SenseVoice/blob/main/model.py) + [README](https://github.com/FunAudioLLM/SenseVoice/blob/main/README.md)

| raw（模型輸出） | label（內部表示） | display（CC 副詞） | token ID |
|----------------|-----------------|-------------------|----------|
| `HAPPY` | `Happy` | `happily` | 25001 |
| `SAD` | `Sad` | `sadly` | 25002 |
| `ANGRY` | `Angry` | `angrily` | 25003 |
| `NEUTRAL` | `Neutral` | （省略） | 25004 |
| `FEARFUL` | `Fearful` | `fearfully` | — |
| `DISGUSTED` | `Disgusted` | `with disgust` | — |
| `SURPRISED` | `Surprised` | `with surprise` | — |
| `unk` | `Unknown` | （省略） | 25009 |

> **注意**：SenseVoice README 列出 7 種情緒，但 `model.py` 的 `emo_dict` 只映射 5 個 token（happy/sad/angry/neutral/unk）。實際 Small 模型可能只輸出這 5 類，需實測確認。

**SenseVoice 音訊事件標籤**：

SenseVoice 除了情緒外，還輸出音訊事件標籤，可納入轉錄輸出：

| raw（模型輸出） | display（CC 慣例） | 說明 |
|----------------|-------------------|------|
| `Speech` | （省略，預設） | 正常語音 |
| `BGM` | `[background music]` | 背景音樂 |
| `Applause` | `[applause]` | 掌聲 |
| `Laughter` | `[laughter]` | 笑聲 |
| `Cry` | `[crying]` | 哭聲 |
| `Sneeze` | `[sneeze]` | 噴嚏 |
| `Breath` | `[breathing]` | 呼吸聲 |
| `Cough` | `[cough]` | 咳嗽 |

**Emotion2vec+ Large 情緒標籤**（可選 SER 模型，9 類）：

來源：[HuggingFace emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)

| index | raw（模型輸出） | label（內部表示） | display（CC 副詞） |
|-------|----------------|-----------------|-------------------|
| 0 | `angry` | `Angry` | `angrily` |
| 1 | `disgusted` | `Disgusted` | `with disgust` |
| 2 | `fearful` | `Fearful` | `fearfully` |
| 3 | `happy` | `Happy` | `happily` |
| 4 | `neutral` | `Neutral` | （省略） |
| 5 | `other` | `Other` | （省略） |
| 6 | `sad` | `Sad` | `sadly` |
| 7 | `surprised` | `Surprised` | `with surprise` |
| 8 | `unknown` | `Unknown` | （省略） |

> **注意**：Emotion2vec+ 的 raw 輸出為小寫（`happy`），SenseVoice 為大寫（`HAPPY`）。內部 `label` 層統一為首字母大寫。

**輸出規則**：
- **TXT / SRT**：使用 `display` 欄位（CC Manner Caption），Neutral/Unknown/Other/Speech 省略
- **JSON**：同時包含 `raw`、`label`、`display` 三層，供程式處理

### TXT（預設格式）

簡潔的純文字逐字稿格式，類似會議記錄。同一講者連續發言時合併為一段，情緒標記參考 CC Manner Caption 慣例，只在變化時標註。

```
CEO_Wang [happily]
今天的季度數據非常亮眼，團隊辛苦了。我們接下來看看各部門的表現。
好的，我們討論 B 事業部的改善方案。

Manager_Lin
是的，我來報告一下各部門的具體數字。Q1 整體營收成長 15%，其中 A 事業部貢獻最大。

speaker_1 [angrily]
但是我想指出，B 事業部的問題還是沒有解決。上個月我們已經討論過了，進度完全沒有推進。
我建議我們重新檢視目前的策略。
```

**標記規範**（參考 CC Manner Caption 慣例）：

- **講者**：獨立一行，講者名稱直接顯示（不加括號）
- **情緒**：`[adverb]` — CC Manner Caption 副詞形式，接在講者名稱後方或情緒變化處
- **Neutral 省略** — CC 慣例只標「非顯而易見」的語氣
- **情緒變化**：同一講者發言中情緒改變時，在變化處獨立插入 `[新情緒]`

**合併規則**：

1. **講者合併**：同一講者連續發言的多個 segment 合併為一段文字
2. **講者切換**：不同講者之間以空行分隔
3. **情緒未變化**：連續相同情緒的文字間不重複標註

### JSON

```json
{
  "metadata": {
    "file": "meeting.wav",
    "duration": "01:02:30",
    "speakers_detected": 4,
    "speakers_identified": 3,
    "date": "2026-04-04"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 15.2,
      "speaker": "CEO_Wang",
      "emotion": {
        "raw": "HAPPY",
        "label": "Happy",
        "display": "happily"
      },
      "audio_event": "Speech",
      "language": "zh",
      "text": "今天的季度數據非常亮眼，團隊辛苦了",
      "confidence": {
        "speaker": 0.92,
        "emotion": 0.87
      }
    }
  ]
}
```

### SRT（DCMP CC 字幕慣例）

採用 DCMP（Described and Captioned Media Program）無障礙字幕標準：
- **講者**：`(Name)` — DCMP 標準括號格式
- **情緒**：`[adverb]` — CC Manner Caption 標準，用副詞形式
- **Neutral 省略** — CC 慣例只標「非顯而易見」的語氣

```
1
00:00:00,000 --> 00:00:15,200
(CEO_Wang) [happily] 今天的季度數據非常亮眼，團隊辛苦了

2
00:00:15,200 --> 00:00:32,000
(Manager_Lin) 是的，我來報告一下各部門的具體數字

3
00:00:32,000 --> 00:00:45,000
(speaker_1) [angrily] 但是我想指出，B 事業部的問題還是沒有解決
```

情緒標籤對應見上方「情緒字串對應表」。

---

## 10. 模型準備與分發

### 分發策略：首次執行自動下載

使用者不需要手動下載模型。Binary 首次執行時自動從網路下載到全局目錄：

```
$ metr transcribe --input meeting.wav

Models not found. Downloading to ~/.metr/models/...
  ggml-large-v3.bin              [████████████████████] 3.0GB ✓
  campplus_sv_zh-cn.onnx         [████████████████████]  28MB ✓
  sensevoice-small-int8.tar.bz2  [████████████████████] 228MB ✓

All models ready. Starting transcription...
```

### 模型清單與來源

| 模型 | 大小 | 格式 | 來源 | 狀態 |
|------|------|------|------|------|
| Whisper Large-v3 | ~3GB | GGML | HuggingFace（ggerganov） | ✓ 現成（`auto`/`en` 預設） |
| Breeze ASR 25 | ~1GB (q5_k) | GGML | HuggingFace（社群轉換） | ✓ 現成（`zh-TW` 自動選擇） |
| Belle-whisper-large-v3-turbo-zh | ~1.5GB | GGML | HuggingFace（BELLE-2） | ✓ 現成（`zh` 自動選擇） |
| kotoba-whisper-v2.0 | ~1.4GB | GGML | HuggingFace（kotoba-tech） | ✓ 現成（`ja` 自動選擇） |
| CAM++ zh-cn | ~28MB | ONNX | GitHub（sherpa-onnx releases） | ✓ 現成可下載（192-dim） |
| SenseVoice-Small (int8) | ~228MB tar.bz2 | ONNX | sherpa-onnx releases | ✓ 現成可下載（**預設 SER**） |
| Emotion2vec+ Large | ~300MB | ONNX | 需自行轉換後上傳 | ⚠ 可選 SER，見下方風險說明 |
| Diarization (FluidAudio) | — | CoreML | ~/.metr/models/diarization/（自動下載） | ✓ metr-diarize 管理 |

### Whisper Large-v3 (GGML) — 現成可下載

```
來源：https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
大小：~3GB
格式：GGML（whisper.cpp 原生格式）
```

### Breeze ASR 25 (GGML) — 台灣中文推薦，現成可下載

```
來源：https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp
大小：~3GB（原始）/ ~1GB（q5_k 量化）
格式：GGML（whisper.cpp 原生格式，社群轉換）
授權：Apache 2.0（可商用）
```

聯發創新基地（MediaTek Research）基於 Whisper-large-v2 微調，專為**台灣華語**和**中英混合（code-switching）**優化。直接輸出繁體中文，解決 Whisper v3 常輸出簡體中文的問題。

可用量化版本：

| 來源 | 量化格式 |
|------|---------|
| [alan314159/Breeze-ASR-25-whispercpp](https://huggingface.co/alan314159/Breeze-ASR-25-whispercpp) | q4_k, q5_k, q8_0 |
| [lsheep/Breeze-ASR-25-ggml](https://huggingface.co/lsheep/Breeze-ASR-25-ggml) | q4_0, q5_0 |

### Taiwan Tongues ASR CE — 政府專案，未來可選

```
來源：https://github.com/adi-gov-tw/Taiwan-Tongues-ASR-CE
格式：HuggingFace Transformers（需轉換為 GGML）
授權：MIT
支援語言：國語、台語、客語、英語
主導：數位發展部數位產業署 / 台灣大哥大
```

目前無現成 GGML 版本，但可用 whisper.cpp 內建的 `convert-h5-to-ggml.py` 轉換。支援台語和客語是其獨特優勢。

### ASR 模型選擇指南

使用者只需設定 `language`，模型自動選擇：

| 設定 | 自動下載的模型 | 大小 | 適合場景 |
|------|--------------|------|---------|
| `language: zh-TW` | Breeze ASR 25 q5_k | ~1GB | 台灣口音、繁體直出、中英混合 |
| `language: zh` | Belle-whisper-large-v3-turbo-zh | ~1.5GB | 簡體中文、中國口音 |
| `language: ja` | kotoba-whisper-v2.0 | ~1.4GB | 日文特化 |
| `language: auto` | Whisper Large-v3 | ~3GB | 多語言自動偵測（99 種語言） |
| **需要台語/客語** | Taiwan Tongues（需手動轉換 GGML） | — | 唯一支援台灣本土語言 |

### CAM++ zh-cn (ONNX) — 現成可下載

```
來源：https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common_advanced.onnx
大小：~30MB
格式：ONNX（onnxruntime-go 直接載入）
下載後重命名為：campplus_sv_zh-cn.onnx（與 config 預設路徑一致）
```

### SenseVoice-Small int8 (ONNX) — 預設 SER 模型，現成可下載

```
來源：sherpa-onnx releases（sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2）
大小：~66MB（int8 量化版）
格式：ONNX（onnxruntime-go 直接載入）
授權：FunASR Model License（允許商用）
情緒分類：最多 7 類 + unk（見 §9 情緒對應表）
```

> **注意**：SenseVoice 推理時會同時產出 ASR + SER 結果。本工具只取 SER 情緒標籤，ASR 仍由 whisper.cpp 負責（精度更高）。

### Emotion2vec+ Large (ONNX) — 可選 SER 模型，需自行轉換

**現況**：官方只提供 PyTorch 格式，無現成 ONNX 版本。需由開發者預先轉換後上傳。
**授權**：CC BY-NC 4.0（非商業用途）。商用場景請使用預設的 SenseVoice。

**轉換步驟**（開發者執行，非使用者）：

```bash
# 1. 安裝依賴
pip install funasr onnx onnxruntime

# 2. 匯出 ONNX
python scripts/export_emotion2vec.py \
  --model iic/emotion2vec_plus_large \
  --output emotion2vec_plus_large.onnx

# 3. 驗證 ONNX 模型
python -c "import onnxruntime; sess = onnxruntime.InferenceSession('emotion2vec_plus_large.onnx'); print('OK')"

# 4. 上傳到 GitHub Releases 或 HuggingFace
```

**已知風險**：
- FunASR 的 ONNX export 對 emotion2vec 的支援不完整（[FunASR #2291](https://github.com/modelscope/FunASR/issues/2291)）
- 社群回報 conv1d 動態 shape 轉換問題（[emotion2vec #55](https://github.com/ddlBoJack/emotion2vec/issues/55)）
- 可能需要修改 export 腳本處理維度問題

**備案**（若轉換失敗）：
1. 使用 Emotion2vec+ Base（較小模型，轉換可能更簡單）
2. 維持 SenseVoice-Small 作為唯一 SER 模型（已是預設，無需額外依賴）
3. SER 9 類功能延後至 Emotion2vec+ ONNX 轉換問題解決

### 模型託管位置

| 平台 | 單檔限制 | 費用 | 用途 |
|------|---------|------|------|
| **GitHub Releases** | 2GB | 免費，無流量限制 | **首選**：Emotion2vec+ ONNX 轉換後上傳 |
| **HuggingFace** | 500GB | 免費 | Whisper 模型來源 |
| sherpa-onnx releases | — | — | CAM++ 模型來源 |

### 全局模型目錄

```
~/.metr/
├── bin/
│   ├── whisper-cli                    # whisper.cpp CLI Binary
│   ├── ffmpeg                         # 音訊格式轉換
│   └── metr-diarize                   # FluidAudio Swift CLI
└── models/
    ├── ggml-large-v3.bin              # Whisper (~3GB, auto/en 預設)
    ├── ggml-breeze-asr-25-q5k.bin     # Breeze ASR 25 (~1GB, zh-TW)
    ├── ggml-belle-zh.bin              # Belle-whisper (~1.5GB, zh)
    ├── ggml-kotoba-whisper-v2.0.bin   # kotoba-whisper (~1.4GB, ja)
    ├── silero-vad-v6.2.0.onnx         # VAD
    ├── campplus_sv_zh-cn.onnx         # CAM++ (~28MB, 192-dim)
    ├── sensevoice-small-int8/         # SenseVoice-Small int8 (~228MB tar.bz2, 預設 SER)
    └── diarization/                   # FluidAudio CoreML 模型（metr-diarize 管理）
```

---

## 11. 編譯與打包

### 編譯流程總覽（Makefile 驅動）

```
Step 1: 編譯 metr-diarize Swift CLI
  make swift
  cd tools/metr-diarize && swift build -c release
  → embedded/binaries/darwin-arm64/metr-diarize

Step 2: 編譯 whisper.cpp CLI
  make whisper
  scripts/build-whisper.sh
  → embedded/binaries/darwin-arm64/whisper-cli

Step 3: 取得 ffmpeg
  make ffmpeg
  → embedded/binaries/darwin-arm64/ffmpeg

                    ↓ 三者就位後 ↓

Step 4: 編譯 Go Binary（embed.FS 自動嵌入，無 build tag）
  make build
  go build → metr
  內含 whisper-cli + ffmpeg + metr-diarize
```

### Step 1: 編譯 metr-diarize Swift CLI

```bash
# Makefile target: swift
cd tools/metr-diarize
swift build -c release
cp .build/release/metr-diarize ../../embedded/binaries/darwin-arm64/
```

### Step 2: 編譯 whisper.cpp CLI

```bash
# scripts/build-whisper.sh
# 自動偵測平台，編譯 whisper.cpp 為獨立 CLI

#!/bin/bash
set -e

WHISPER_VERSION="v1.7.3"  # 鎖定版本
OUTDIR="embedded/binaries/darwin-arm64"

# clone + cmake build（Metal GPU 加速）
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=1 \
  -DGGML_METAL_EMBED_LIBRARY=1 \
  -DBUILD_SHARED_LIBS=0

cmake --build build -j --config Release --target whisper-cli
mkdir -p "$OUTDIR"
cp build/bin/whisper-cli "$OUTDIR/"
```

### Step 3 & 4: Go Binary 編譯

```go
// embedded/embedded.go
package embedded

import "embed"

//go:embed binaries/darwin-arm64/whisper-cli
//go:embed binaries/darwin-arm64/ffmpeg
//go:embed binaries/darwin-arm64/metr-diarize
var Binaries embed.FS

const CacheDir = ".metr"

// ExtractAll 首次執行時解壓到 ~/.metr/bin/
// 比對嵌入的 hash，若快取已存在且一致則跳過
func ExtractAll() (binDir string, err error) {
    // 1. 建立 ~/CacheDir/bin/
    // 2. 寫入 whisper-cli（設定 0755 權限）
    // 3. 寫入 ffmpeg（設定 0755 權限）
    // 4. 寫入 metr-diarize（設定 0755 權限）
    // 5. 回傳 binDir 路徑
}
```

```bash
# 最終編譯
CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 \
  go build -o metr ./cmd/main.go
```

### Makefile 整合

```makefile
.PHONY: all build swift whisper ffmpeg clean clean-all info

# 完整建置：Swift CLI + whisper-cli + ffmpeg + Go binary
all: swift whisper ffmpeg build

# 編譯 metr-diarize Swift CLI（FluidAudio CoreML/ANE）
swift:
	cd tools/metr-diarize && swift build -c release
	cp tools/metr-diarize/.build/release/metr-diarize embedded/binaries/darwin-arm64/

# 編譯 whisper.cpp CLI（Metal GPU）
whisper:
	bash scripts/build-whisper.sh

# 取得 ffmpeg
ffmpeg:
	bash scripts/build-ffmpeg.sh

# 編譯 Go Binary（embed.FS 自動嵌入 3 個 Binary，無 build tag）
build:
	CGO_ENABLED=1 GOOS=darwin GOARCH=arm64 \
	  go build -o metr ./cmd/main.go

clean:
	rm -f metr

clean-all:
	rm -f metr
	rm -rf embedded/binaries/
	go clean -cache

info:
	@echo "Platform: $(shell uname -s)-$(shell uname -m)"
	@ls embedded/binaries/darwin-arm64/ 2>/dev/null || echo "No binaries built yet"
```

### 交付物

```
metr-v0.1.0-darwin-arm64/
├── metr          # Go Binary（內含 whisper-cli + ffmpeg + metr-diarize）
└── README.md
# Binary 本體 ~80MB（含嵌入的 3 個 Binary）
# 模型不隨 Binary 分發，首次執行時自動下載（~3+GB）
```

### 安裝流程

```
1. 下載 Binary → 放入 PATH（或任意位置）
2. 執行 `metr init` → 建立 speakers/ + output/
3. 首次 transcribe 時自動下載模型到 ~/.metr/models/
```

> **設計原則**：使用者拿到的只有一個執行檔。模型自動下載、依賴自動解壓，零手動設定。

---

## 12. 驗證計畫

### 單元測試

| 模組 | 測試內容 |
|------|---------|
| `audio` | WAV 讀取、重採樣、片段切割 |
| `speaker/matcher` | Cosine similarity 計算、門檻值判定 |
| `speaker/store` | 資料夾掃描、hash 計算、.profile.json 讀寫、快取失效判定 |
| `output/*` | 各格式輸出正確性 |

### 整合測試

1. **Enrollment E2E**：
   - 建立 `speakers/TestUser/` 放入測試 WAV
   - 執行 `enroll` → 確認 `.profile.json` 已產生且內容正確
   - 新增一個 WAV → 再次 `enroll` → 確認 status 為 "updated"
   - 不修改 → 再次 `enroll` → 確認 status 為 "unchanged"（快取命中）
   - 刪除一個 WAV → 再次 `enroll` → 確認 embedding 已重算
2. **Verify E2E**：`speakers verify --name TestUser --audio test.wav` → 確認回傳 similarity 分數
3. **Transcribe E2E**：提供多講者測試音檔 → transcribe → 驗證輸出包含正確的 speaker / emotion / text
4. **Auto-Enroll E2E**：刪除 `.profile.json` → 直接執行 `transcribe` → 確認自動重建快取
5. **Diarization + Speaker Resolve E2E**：
   - 使用包含多位講者的測試音檔 → transcribe → 確認 metr-diarize 子程序正常執行
   - 確認 ASRResult 與 DiarSegment 正確依時間重疊合併
   - 已註冊講者的片段確認標註正確姓名
   - 未註冊講者的片段確認以 `speaker_N` 標註
6. **Binary 測試**：`make all` 後確認 `embedded/binaries/darwin-arm64/` 含 3 個 Binary；`otool -L metr` 確認無意外系統動態庫依賴

### 測試音檔

準備以下測試素材：
- 單一講者 15 秒清晰語音 × 3 人（放入 `speakers/` 資料夾，用於 enrollment）
- 另一段各講者的短音檔（用於 `speakers verify` 測試）
- 多講者對話 2-3 分鐘（用於 transcribe 測試）
- 包含明顯情緒變化的片段（驗證 SER）
