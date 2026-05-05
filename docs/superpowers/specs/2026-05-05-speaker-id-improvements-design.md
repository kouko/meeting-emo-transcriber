# Speaker Identification 4-Fix 設計規格

> 把當前憑魔術數字 threshold 的二元 matcher，升級成統計校準 + filtered centroid + continuous enrollment + 三段信心輸出的決策 pipeline。

## 1. 背景與問題

### 1.1 當前 pipeline 狀態（PR #14 之後）

`internal/diarize/merge.go ResolveSpeakerNames` 目前流程：

```
1. metr-diarize 產出 cluster + per-cluster centroid（FluidAudio 不透明算法）
2. 對每個 cluster：
   a. 取 diarResult.SpeakerVoiceprints[clusterID] 作為比對 input
   b. 對每個 enrolled profile 跑 cosine similarity（max-over-templates）
   c. 取最高 sim 對應的 profile name
   d. if bestSim >= match_threshold (0.55)：matched
   e. else if longest segment >= min_sample_duration (15s)：建 speaker_N/ + persist 樣本
   f. else：標 "Unknown"
3. 把 cluster name map 回 ASR segments
```

### 1.2 觀察到的失敗模式

跨 session 的 1:N speaker identification 在以下情境表現不穩：

1. **Threshold 不適配 vault**：`match_threshold = 0.55` 是硬編碼預設值；不同 vault 的 enrolled-vs-enrolled 與 enrolled-vs-other 分布不同（人數、性別比、語言、麥克風通道），同一 threshold 對某 vault 過鬆對另一 vault 過嚴
2. **Centroid 不可審計**：`diarResult.SpeakerVoiceprints[clusterID]` 是 FluidAudio 內部 `result.speakerDatabase` 算出來的；無法確認它有否 filter 短 segment / overlap speech / 低品質 segment
3. **Profile 凍結**：enrollment 時固定，無法反映 voice drift（時段、感冒、麥克風更換）
4. **二元決策資訊壓縮**：cluster 不是 match 就是 unknown；中間區段（cosine 0.50-0.70）的不確定性對下游不可見

### 1.3 不在本次 scope 的問題

以下需要更換 embedding model 才能根本解決，**不在本 spec 範圍**：

- WeSpeaker 256d 區辨力天花板（升級到 ECAPA2 768d 或 pyannote 4.0 community-1）
- VoxCeleb domain gap（台灣國語 / 國台語混講）
- Cluster 本身錯誤（FluidAudio diarize 把 Alice 部分歸給 Bob 的 cluster）

本 spec 處理「在現有 model 之上榨出最大值」的 4 個工程改進。

## 2. 4 個修正概覽

### 2.1 Fix 1：EER threshold 自動校準

**動機**：`match_threshold` 改用統計校準的 EER (Equal Error Rate) 點，而非經驗值 0.55。

**核心算法**：對 enrolled profiles 算兩個分布——

| 分布 | 計算 |
|---|---|
| Genuine | 同一 speaker 內部 voiceprints 兩兩配對的 cosine sim |
| Impostor | 跨 speaker 的 voiceprints 兩兩配對的 cosine sim |

找 EER 點 = FAR (false accept rate) ≡ FRR (false reject rate) 的 cosine 值，寫回 config.yaml。

**新 subcommand**：`metr calibrate [--apply] [--target-far <float>] [--output <path>]`

### 2.2 Fix 2：Filter + mean-pool centroid 重算

**動機**：取代 FluidAudio centroid。在 Go 端顯式：

1. Filter cluster 的 segments：≥1s + 非 overlap + RMS ≥ min_sample_rms（重用 PR #14 的 filter）
2. 切音 + batch 呼叫 `metr-diarize --extract-voiceprints`（已存在 mode）
3. Mean-pool 結果作為 centroid

**Fallback**：如果某 cluster 沒有合格 segment，退回 FluidAudio centroid + warning。

### 2.3 Fix 3：High-confidence continuous enrollment

**動機**：profile drift 自癒。matched 且 sim > confident_threshold 的 cluster，自動寫代表 audio 進 `~/metr-speakers/{name}/auto/`，下次 AutoEnroll 自動 absorb。

**Pruning**：FIFO 限 N 個（預設 10）。新樣本 > N 時刪最舊。

**Drift detection**：每次 AutoEnroll 後，新 voiceprint vs 既有 voiceprint cosine < 0.85 → 印 warning。

### 2.4 Fix 4：3-tier confidence band 輸出

**動機**：把不確定性顯式化。把 cluster 名稱對應從二值改三段：

| Band | 範圍 | 顯示 |
|---|---|---|
| Confident | sim >= confident_threshold (EER + 0.15) | `Alice` |
| Tentative | EER <= sim < confident_threshold | `Alice?`（可選；JSON 永遠帶 confidence）|
| Unknown | sim < EER | `speaker_N` |

JSON 輸出 additive，加 `confidence.speaker_band` + `confidence.speaker_candidates`（top-K）。

## 3. 與 PR #14 的關係

PR #14（`fix: filter short/silent segments from auto-discovered speaker samples`）已經在 **enrollment side** 加了：
- `min_sample_duration = 15s`（cluster level：longest segment 過短不建 speaker_N/）
- `min_sample_rms = 0.01`（per-WAV silence filter）
- `audio.RMS()` utility

本 spec 的 Fix 2 是 **identification side** 的對應改動：
- 處理階段：matched cluster 對 enrolled profile 比對時，centroid 怎麼算
- 新 thresholds：`min_centroid_segment_duration` (~1-3s，比 enrollment 的 15s 鬆) + `exclude_overlap`
- Reuse `audio.RMS()` 作為 segment-level filter

兩者獨立 logic 路徑，不衝突。

## 4. 詳細設計

### 4.1 Fix 1：EER calibration

#### 4.1.1 算法

```go
// internal/speaker/calibrate.go

// ComputeDistributions 對所有 enrolled profiles 算 genuine + impostor 分布。
// genuine: 同 profile 內 voiceprints 兩兩 (N choose 2)
// impostor: 跨 profile voiceprints 兩兩 (M-N) × N
func ComputeDistributions(profiles []types.SpeakerProfile) (genuine, impostor []float32, err error)

// FindEER 找 FAR == FRR 的點。
// 排序 genuine 升序 + impostor 降序，掃 threshold 從低到高。
// 在每個候選點計算：
//   FRR = #{genuine < threshold} / len(genuine)
//   FAR = #{impostor >= threshold} / len(impostor)
// EER = 第一個 FAR <= FRR 的 threshold（線性插值更精確）
func FindEER(genuine, impostor []float32) (threshold, eer float32)

// CalibrationReport 印 ASCII histogram + EER 點 + 建議 thresholds
func CalibrationReport(genuine, impostor []float32, eer float32) string
```

#### 4.1.2 CLI

```bash
# 印報告，不寫回 config
metr calibrate

# 寫回 config.yaml（match_threshold + 衍生值）
metr calibrate --apply

# 用更嚴的 FAR 而非 EER 點
metr calibrate --target-far 0.01 --apply

# 輸出報告檔
metr calibrate --output calibration-report.txt
```

#### 4.1.3 Config 變化

```yaml
# 新欄位
calibrated_eer: 0.58              # 校準的 EER 點
calibration_eer_value: 0.06       # EER 值（FAR == FRR 時的錯誤率）
calibrated_at: "2026-05-05"       # 上次校準時間
calibrated_with_n_profiles: 5     # 校準時 vault 大小

# 既有欄位行為改變（已 deprecated 但向下相容）
match_threshold: 0.55             # 若 calibrated_eer 存在則被 override
```

#### 4.1.4 Edge cases

- Vault < 3 speakers 或某 speaker < 2 voiceprints → genuine 樣本不足，**警告 + 維持預設 0.55**
- Genuine / impostor 完全分離（0% EER）→ 取 genuine 最低點 - 0.05
- 完全重疊（高 EER）→ 警告「embedding model 區辨力不足」
- **不 mutate config 除非 `--apply`**（避免 surprise）

#### 4.1.5 預估規模

~250 LOC + ~150 LOC test。1-2 commit。

### 4.2 Fix 2：Filter + mean-pool centroid

#### 4.2.1 流程

```go
// internal/diarize/merge.go ResolveSpeakerNames 內

for _, clusterID := range clusterIDs {
    // 既有：取 FluidAudio centroid
    fluidCentroid := diarResult.SpeakerVoiceprints[clusterID]

    // 新：filter + recompute
    filtered := filterClusterSegments(diarResult.Segments, clusterID, wavSamples, sampleRate, cfg)

    var centroid []float32
    if len(filtered) >= cfg.MinSegmentsForRecompute {
        // 切音 → 寫 temp wav
        wavPaths := writeTempWavs(filtered, wavSamples, sampleRate)
        defer cleanupTemp(wavPaths)

        // batch extract
        results, err := diarize.ExtractVoiceprints(diarizeBinPath, wavPaths)
        if err != nil {
            // fallback
            centroid = float64sToFloat32s(fluidCentroid)
        } else {
            centroid = meanPool(results)
        }
    } else {
        // fallback: 不夠 segment → 用 FluidAudio centroid
        centroid = float64sToFloat32s(fluidCentroid)
    }

    // 後續 match 邏輯不變
    result := matchAgainstProfilesDetailed(centroid, profiles, threshold)
    ...
}
```

#### 4.2.2 Filter 條件

```go
type CentroidFilter struct {
    MinDuration       float64  // default 1.5s
    ExcludeOverlap    bool     // default true
    MinRMS            float64  // default 0.01 (重用 PR #14)
    MaxSegmentsPerCluster int   // default 10 (避免長會議慢)
}

// 過濾：duration >= MinDuration && 不 overlap 其他 cluster && RMS >= MinRMS
// 排序：duration 大到小，取 top MaxSegmentsPerCluster
```

#### 4.2.3 Config 變化

```yaml
# 新增
min_centroid_segment_duration: 1.5
exclude_overlap_in_centroid: true
max_segments_per_cluster: 10
min_segments_for_recompute: 2     # 少於此數，fallback FluidAudio centroid
```

#### 4.2.4 Performance

- 每 cluster ~5 個 1-2s segments
- `metr-diarize --extract-voiceprints` batch mode：model load 1 次 + N forward passes
- 預估每 cluster ~250-500ms 額外耗時
- 30 cluster meeting → ~7.5-15s overhead（< 1% of typical Whisper time）

#### 4.2.5 預估規模

~350 LOC（含 wav 切音 helper）+ ~200 LOC test。3-4 commit。

### 4.3 Fix 3：Continuous enrollment

#### 4.3.1 流程

```go
// 在 ResolveSpeakerNames matched 分支，bestSim > confidentThreshold 時：

if confidentThreshold > 0 && result.BestSim > confidentThreshold {
    // 取 cluster 內最長的 segment 作代表
    bestSeg := pickBestSegment(diarResult.Segments, clusterID)
    audioBytes := extractWavBytes(wavSamples, sampleRate, bestSeg)

    speaker.PersistContinuousSample(
        store,
        result.Name,
        audioBytes,
        result.BestSim,
        cfg.ContinuousEnrollmentMaxSamples,
    )
}
```

#### 4.3.2 檔案布局

```
~/metr-speakers/Alice/
├── enroll_001.wav                # 既有
├── alice.profile.json            # 既有
└── auto/                         # 新
    ├── 20260505_142308_sim083.wav
    ├── 20260507_091233_sim079.wav
    └── 20260510_153012_sim085.wav
```

`AutoEnroll` 認得 `auto/` 子目錄並含進 enrollment 的 wav concat。

#### 4.3.3 Config 變化

```yaml
continuous_enrollment: true
continuous_enrollment_max_samples: 10
continuous_enrollment_threshold_margin: 0.15   # threshold = EER + margin
```

#### 4.3.4 Drift detection

```go
// AutoEnroll 後比對新舊 voiceprint
oldVoiceprint := mostRecentVoiceprint(profile)
newVoiceprint := computedFromMergedWav

driftSim := CosineSimilarity(oldVoiceprint, newVoiceprint)
if driftSim < cfg.DriftWarnThreshold {  // default 0.85
    log.Warnf("Profile %s drifted significantly (cosine %.2f vs previous)", name, driftSim)
}
```

#### 4.3.5 Edge cases

- **同 cluster 多個 high-conf segment**：只存 1 個（cluster 級代表 audio）
- **Hash collision**：既有 `KnownAudioHashes` 機制，AutoEnroll 跳過重複
- **誤差雪球**：threshold_margin 要嚴（0.15+）；如果 false-positive 進 auto/ 會污染 profile
- **隱私**：auto/ 樣本是會議 audio。提供 `--no-continuous-enrollment` flag for 敏感場合

#### 4.3.6 預估規模

~200 LOC + ~150 LOC test。1-2 commit。

### 4.4 Fix 4：3-tier confidence band

#### 4.4.1 內部結構

```go
// internal/types/types.go 修改

type Confidence struct {
    Speaker            float32     `json:"speaker"`             // 既有
    SpeakerBand        string      `json:"speaker_band"`        // 新："confident" | "tentative" | "unknown"
    SpeakerCandidates  []Candidate `json:"speaker_candidates,omitempty"` // 新：top-K
    Emotion            float32     `json:"emotion"`             // 既有
}

type Candidate struct {
    Name       string  `json:"name"`
    Similarity float32 `json:"similarity"`
}
```

#### 4.4.2 分類邏輯

```go
func classifyConfidence(sim, eerThreshold, confidentThreshold float32) string {
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

#### 4.4.3 輸出格式

| 格式 | Confident | Tentative | Unknown |
|---|---|---|---|
| TXT（預設） | `Alice` | `Alice` （不標 ?） | `speaker_3` |
| TXT (`--show-confidence`) | `Alice [conf=0.83]` | `Alice? [conf=0.62]` | `speaker_3 [conf=0.41]` |
| JSON | 完整 confidence struct | 完整 confidence struct | 完整 confidence struct |
| SRT | `Alice` | `Alice` | `speaker_3` |

**為什麼 TXT 預設不標 `?`**：保留 user 既有閱讀習慣；不確定性留在 JSON 給 LLM wrapper 處理。

#### 4.4.4 Config 變化

```yaml
# 衍生欄位（從 calibrated_eer 計算，user 不直接設）
tentative_threshold: 0.58          # = calibrated_eer
confident_threshold: 0.73          # = calibrated_eer + 0.15
show_confidence_in_txt: false      # default
```

#### 4.4.5 預估規模

~150 LOC + ~100 LOC test。1 commit。

## 5. 執行順序與依賴

```
Fix 1 (calibrate)
    ├──→ Fix 4 (confidence band)        [需 thresholds]
    └──→ Fix 3 (continuous enrollment)  [需 confident_threshold]

Fix 2 (mean-pool centroid)
    獨立                                 [改 ResolveSpeakerNames 的 input]
```

| Order | Fix | 預估 LOC（含 test）| Commit | 預估時間 |
|---|---|---|---|---|
| 1 | Fix 2 (mean-pool centroid) | ~550 | 3-4 | ~1 day |
| 2 | Fix 1 (calibrate) | ~400 | 1-2 | ~half day |
| 3 | Fix 4 (confidence band) | ~250 | 1 | ~half day |
| 4 | Fix 3 (continuous enrollment) | ~350 | 1-2 | ~half day |

**總計 ~1500 LOC，~2-3 個 working day。建議分 4 個 PR**（一 fix 一 PR），不 stack。

## 6. 與 skill wrapper 的分工

本 spec 處理 **binary 層** 的 4 個改進。獨立的 `meeting-toolkit` Claude Code skill（在 monkey-skills repo）會處理：

- Tentative band LLM 自動 resolve（語言指紋 + calendar prior）
- Auto-explain（取代 `metr explain` subcommand）
- Cross-meeting trend tracking
- Action item 抽取
- Privacy mode 自動偵測（從 calendar title 判讀）
- Vault hygiene 主動 alert

**分工原則**：

| 屬性 | binary | skill |
|---|---|---|
| Signal processing（音訊、ASR、diarization） | ✅ | ❌ |
| 統計計算（EER、cosine sim） | ✅ | ❌ |
| File system / config | ✅ | ❌ |
| Confidence band 分類（deterministic） | ✅ | ❌ |
| LLM reasoning（語言指紋、context resolution） | ❌ | ✅ |
| 跨會議推理 | ❌ | ✅ |
| Calendar / Obsidian / 外部整合 | ❌ | ✅ |

→ skill 不取代 binary；skill 在 binary 之上加智能層。

## 7. UX 負擔評估

加 4 個 fix **本身**會給 user 帶來新負擔（calibrate 要記得跑、3 個 threshold 概念、auto/ 要監督等）。Mitigation 設計：

### 7.1 Binary 層 mitigation（本 spec 涵蓋）

- **Threshold 隱藏複雜度**：對 user 來說只 1 個 `calibrated_eer`；`tentative_threshold` / `confident_threshold` 是衍生量不暴露
- **TXT 預設不標 `?`**：保持既有閱讀體驗
- **Auto-detect calibration drift**：vault 大幅變化時 stderr 印 nudge，不阻擋執行
- **Vault audit/rollback subcommand**：`metr speakers audit` / `metr speakers rollback`
- **3-mode preset**：`--mode strict|default|adaptive` 一次決策永久套用

### 7.2 Skill 層 mitigation（不在本 spec）

`meeting-toolkit` skill 接手以下負擔：
- Auto-resolve tentative band（user 看不到 `?`）
- Auto-explain why tentative
- Mode 自動切換（從 calendar 判讀）
- Cross-meeting trend 主動觀察
- Vault drift 對話式 alert

→ binary mitigation 已足夠讓 user 不必重度介入；skill 進一步把介入降到接近於零。

## 8. Acceptance criteria

- [ ] Fix 1：`metr calibrate` 可執行；EER 計算正確（unit test 用合成 voiceprint）；config write-back 行為正確
- [ ] Fix 2：cluster centroid 改用 filtered mean-pool；fallback 邏輯正確；新 wav 切音 + temp file cleanup 無 leak
- [ ] Fix 3：`auto/` 子目錄正確建立 + FIFO 限額；AutoEnroll 認得 `auto/`；drift warning 觸發
- [ ] Fix 4：JSON 輸出含 `speaker_band` + `speaker_candidates`；TXT 預設不變；`--show-confidence` flag 行為正確
- [ ] 整體：在合成 vault（≥5 speakers × ≥3 voiceprints each）跑 EER 計算結果穩定
- [ ] 文件：CHANGELOG 更新；example.config.yaml 加新欄位 + 註解；README 加 `metr calibrate` 章節

## 9. 開放問題

1. **`metr-diarize --extract-voiceprints` 對 1-2s 切片是否準確？** — Swift 內部跑完整 diarization pipeline，雖能正常運作但 overhead 大且可能 < 1s 會 reject。需 prototype 測試
2. **`min_centroid_segment_duration` 預設值該多少？** — 提案 1.5s；待實測 calibration vault 確認
3. **Drift detection 的 threshold 0.85 怎麼定？** — 暫用經驗值；calibrate 命令可考慮一併校準此值
4. **Multi-template profile 支援是否要做？** — 目前 AutoEnroll 把多 wav concat → 1 voiceprint。多 template (例：每個 wav 1 voiceprint，match 取 max) 會更 robust 但 code 改動大；建議 v2

## 10. 參考

- 既有 spec：`docs/superpowers/specs/2026-04-05-diarization-refactor-design.md`（雙 pipeline 架構）
- 既有 PR：#14（PR #14 enrollment-side filter，本 spec 的 Fix 2 是 identification-side 對應）
- WeSpeaker model：`fluidaudio_embedding_v1`（256d，封裝在 metr-diarize）
- 對應 monkey-skills repo 的 `meeting-toolkit` plugin（待設立）為 skill 層
