# Phase 4 設計規格：Unknown Speaker Auto-Discovery + Speakers Verify

> Phase 4 of meeting-emo-transcriber — 自動發現未註冊的說話者並建立 profile，加上 speakers verify 驗證命令。

## 1. 背景與目標

### 為什麼做

Phase 3 完成了 speaker identification，但未匹配的 speaker 一律標為 "Unknown"。
真實會議中經常有未事先 enroll 的與會者，需要：
1. 自動發現並區分不同的未知 speaker（speaker_1, speaker_2...）
2. 持久化發現結果，讓下次轉錄自動使用
3. 提供 verify 命令讓使用者驗證 speaker recognition 品質

### 預期成果

Phase 4 完成後：
- 未匹配的 speaker 自動命名為 `speaker_1`, `speaker_2`...（非全部 "Unknown"）
- 自動建立 `speakers/speaker_N/` 資料夾、儲存音訊樣本、產生 `.profile.json`
- 使用者重命名資料夾後下次自動使用新名稱
- `--no-discover` 停用持久化（仍聚類為 Unknown_N，但不建資料夾）
- `speakers verify --name X --audio Y` 可驗證 speaker 辨識品質

### 範圍

| 包含 | 不包含 |
|------|--------|
| Unknown speaker auto-discovery | 長時間會議分段策略 |
| 跨片段聚類（同人同 ID） | Emotion2vec+ 替代模型 |
| --no-discover flag 實作 | Pipeline 平行化 |
| speakers verify 命令 | GUI / Web 介面 |
| Metadata speaker 計數更新 | |

---

## 2. Unknown Speaker Discovery 模組

### 目錄結構

```
internal/speaker/
├── store.go       # Phase 1（不變）
├── matcher.go     # Phase 1（不變）
├── extractor.go   # Phase 3（不變）
├── discovery.go   # Phase 4 新增
└── discovery_test.go  # Phase 4 新增
```

### 資料結構

```go
// unknownSpeaker 代表本次轉錄 session 中發現的一個未知 speaker
type unknownSpeaker struct {
    name      string     // "speaker_1" 或 "Unknown_1"
    embedding []float32  // 第一次遇到時的 embedding（用於後續比對）
}

// Discovery 管理未匹配 speaker 的自動發現與持久化
type Discovery struct {
    store      *Store
    extractor  *Extractor
    matcher    *Matcher
    threshold  float32
    noDiscover bool
    unknowns   []unknownSpeaker
    nextID     int
}
```

### API

```go
// NewDiscovery 建立 discovery 管理器。
// nextID 初始化時掃描 speakers/ 中現有的 speaker_N 資料夾，
// 取最大 N+1 作為起始值，避免與之前 session 的 unknown 衝突。
func NewDiscovery(
    store *Store,
    extractor *Extractor,
    matcher *Matcher,
    threshold float32,
    noDiscover bool,
) *Discovery

// IdentifySpeaker 辨識 speaker：先比對已註冊 profiles，
// 再比對本次 session 的 unknowns，最後建立新 unknown。
// segAudio 和 segStart 用於持久化（存音訊樣本）。
func (d *Discovery) IdentifySpeaker(
    embedding []float32,
    profiles []types.SpeakerProfile,
    segAudio []float32,
    sampleRate int,
    segStart float64,
) (name string, confidence float32)

// Unknowns 回傳本次 session 發現的所有 unknown speakers
func (d *Discovery) Unknowns() []string
```

### IdentifySpeaker 演算法

```
1. matcher.Match(embedding, profiles, threshold)
   ├─ match.Name != "" → return (match.Name, match.Similarity)
   └─ no match ↓

2. 比對本次 session 的 unknowns
   for each unknown in d.unknowns:
       sim := CosineSimilarity(embedding, unknown.embedding)
       if sim >= threshold:
           return (unknown.name, sim)

3. 建立新 unknown
   d.nextID++
   if d.noDiscover:
       name = fmt.Sprintf("Unknown_%d", d.nextID)
       // 不建資料夾、不存樣本
   else:
       name = fmt.Sprintf("speaker_%d", d.nextID)
       // 建立 speakers/speaker_N/
       // 存 auto_segment_<timestamp>.wav
       // 產生 .profile.json

   d.unknowns = append(d.unknowns, unknownSpeaker{name, embedding})
   return (name, 0)
```

### 持久化邏輯（noDiscover=false 時）

```go
func (d *Discovery) persistUnknown(name string, embedding []float32, segAudio []float32, sampleRate int, segStart float64) error {
    // 1. 建立 speakers/<name>/ 資料夾
    speakerDir := filepath.Join(d.store.Root(), name)
    os.MkdirAll(speakerDir, 0755)

    // 2. 儲存音訊樣本
    timestamp := fmt.Sprintf("%04d", int(segStart*100))
    wavPath := filepath.Join(speakerDir, fmt.Sprintf("auto_segment_%s.wav", timestamp))
    audio.WriteWAV(wavPath, segAudio, sampleRate)

    // 3. 計算 hash 並產生 .profile.json
    hash, _ := FileHash(wavPath)
    profile := types.SpeakerProfile{
        Name: name,
        Embeddings: []types.SampleEmbedding{{
            File:      filepath.Base(wavPath),
            Hash:      hash,
            Embedding: embedding,
        }},
        Dim:       d.extractor.Dim(),
        Model:     "campplus_sv_zh-cn",
        CreatedAt: time.Now().Format(time.RFC3339),
        UpdatedAt: time.Now().Format(time.RFC3339),
    }
    return d.store.SaveProfile(profile)
}
```

### Store 需要新增的方法

```go
// Root 回傳 store 的根目錄路徑
func (s *Store) Root() string {
    return s.root
}
```

---

## 3. speakers verify 命令

### 目錄結構

修改現有 `cmd/commands/speakers.go` 的 verify stub。

### 流程

```
speakers verify --name CEO_Wang --audio test.wav
    │
    ├─ 1. embedded.ExtractAll() → bins
    ├─ 2. models.EnsureModel("campplus-sv-zh-cn") → modelPath
    ├─ 3. config.Load() → cfg
    ├─ 4. speaker.NewExtractor(modelPath, threads)
    ├─ 5. store.LoadProfile("CEO_Wang") → profile
    │      （若 profile 不存在 → 錯誤）
    ├─ 6. audio.ConvertToWAV(bins.FFmpeg, audioPath, tempWav)
    ├─ 7. audio.ReadWAV(tempWav) → samples
    ├─ 8. extractor.Extract(samples, sampleRate) → testEmbedding
    ├─ 9. matcher.Match(testEmbedding, [profile], threshold) → result
    └─ 10. 輸出結果
```

### 輸出格式

```
Verifying against CEO_Wang...
  Similarity: 0.89 (threshold: 0.60)
  Result: ✓ MATCH
```

或不匹配時：
```
Verifying against CEO_Wang...
  Similarity: 0.42 (threshold: 0.60)
  Result: ✗ NO MATCH
```

---

## 4. Transcribe 命令整合

### 修改 transcribe.go

替換 Phase 3 的簡單 fallback 為 Discovery：

```go
// 初始化 Discovery（在 extractor 和 matcher 之後）
discovery := speaker.NewDiscovery(store, extractor, matcher, float32(cfg.Threshold), noDiscover)

// 每個 segment
for _, r := range results {
    segAudio := audio.ExtractSegment(wavSamples, wavSampleRate, r.Start, r.End)

    // Speaker identification（透過 Discovery）
    speakerName := "Unknown"
    var speakerConf float32
    if len(segAudio) > 0 {
        emb, embErr := extractor.Extract(segAudio, wavSampleRate)
        if embErr == nil {
            speakerName, speakerConf = discovery.IdentifySpeaker(
                emb, profiles, segAudio, wavSampleRate, r.Start,
            )
        }
    }

    // Emotion 邏輯不變
    ...
}
```

### Metadata 更新

```go
// Discovery 完成後
uniqueSpeakers := make(map[string]bool)
identified := 0
for _, seg := range segments {
    uniqueSpeakers[seg.Speaker] = true
    if seg.Speaker != "Unknown" && !strings.HasPrefix(seg.Speaker, "speaker_") && !strings.HasPrefix(seg.Speaker, "Unknown_") {
        identified++
    }
}

Metadata{
    SpeakersDetected:   len(uniqueSpeakers),
    SpeakersIdentified: identified,  // 已知（非 unknown）的 segment 數
}
```

---

## 5. 新增/修改檔案清單

### 新增

| 檔案 | 用途 |
|------|------|
| `internal/speaker/discovery.go` | Unknown speaker 自動發現 + 聚類 + 持久化 |
| `internal/speaker/discovery_test.go` | Discovery 測試 |

### 修改

| 檔案 | 修改內容 |
|------|---------|
| `internal/speaker/store.go` | 新增 Root() 方法 |
| `cmd/commands/transcribe.go` | 用 Discovery 替代簡單 fallback |
| `cmd/commands/speakers.go` | 實作 verify 命令（替換 stub） |

---

## 6. 驗證計畫

### 單元測試

| 模組 | 測試內容 |
|------|---------|
| `speaker/discovery` | IdentifySpeaker：已知 speaker 回傳正確名稱 |
| `speaker/discovery` | IdentifySpeaker：未知 speaker 建立 speaker_1 |
| `speaker/discovery` | 跨片段聚類：同一 embedding 兩次呼叫回傳相同 speaker_N |
| `speaker/discovery` | noDiscover=true：命名為 Unknown_N，不建資料夾 |
| `speaker/discovery` | noDiscover=false：建立資料夾、存音訊、產 profile |

### 整合測試

1. **Unknown Speaker E2E**：
   - 使用包含未註冊講者的測試音檔 → transcribe → 確認自動建立 `speaker_N/`
   - 確認資料夾內有 `auto_segment_*.wav` 和 `.profile.json`
   - 重新命名 `speaker_1/` → `TestUser/` → 再次 transcribe → 確認使用新名稱

2. **--no-discover E2E**：
   - transcribe --no-discover → 確認不建立新資料夾
   - 輸出中有 Unknown_1, Unknown_2（非全部 Unknown）

3. **speakers verify E2E**：
   - 已 enroll 的 speaker + 同人音檔 → verify → MATCH
   - 已 enroll 的 speaker + 不同人音檔 → verify → NO MATCH

### 手動驗證

- 用多人對話音檔測試 auto-discovery 聚類正確性
- 重命名資料夾後再轉錄驗證
