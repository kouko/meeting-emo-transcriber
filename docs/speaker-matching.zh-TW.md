# 發話者辨識機制（繁體中文）

這份文件用白話解釋 metr 怎麼判斷「這段聲音是誰」。對應的英文設計討論散在
commit message、CHANGELOG.md 與 README.md 的 Speaker Matching
Architecture 區塊。

## 兩步驟流程

```
錄音檔
  ↓
(1) 分群（diarization）：把連續說話的時段切開，標上 cluster_0、cluster_1…
       這一步「不認人」，只負責看出有幾個不同的聲音
  ↓
(2) 認人（matching）：拿每個 cluster 的「聲紋」對照你登錄過的 enrolled
       profiles，決定要不要給名字
```

第 (1) 步由 FluidAudio（pyannote 風格的分割 + WeSpeaker 256 維聲紋）
跑完之後，每個 cluster 都有一個**代表聲紋**（centroid）。

第 (2) 步是這次重點重寫的部分。

## 名詞對照

- **聲紋（voiceprint / embedding）**：一段聲音被模型壓縮成的一串浮點數。同一個人多次說話，聲紋會長得很像。
- **cosine 相似度**：兩個聲紋的「夾角」，-1 ~ 1 之間，越接近 1 表示越像同一個人。
- **登錄（enrollment）**：你把家人/同事的乾淨錄音放進 `<speakers>/<name>/` 資料夾，metr 會把它變成那個人的聲紋存到 `profile.json`。
- **閾值（threshold）**：「至少多像才算同一個人」的及格分數。

## 認人的兩道閘門

第 (2) 步做完所有 cluster × profile 的相似度比較後，每個 cluster 同時要過兩個檢查才算「找到名字」：

1. **基本閾值**：`最高相似度 >= match_threshold`（預設 **0.65**）
2. **差距規則**：`最高相似度 - 第二高相似度 >= match_margin`（預設 **0.07**）

第二個檢查的用意：如果這個 cluster 和 Alice 與 Bob 都很像（分數差不多），那硬選一個有可能是錯的，乾脆都不選，改成 `speaker_N` 給你手動確認。

## 一對一指派（Hungarian-style）

舊版本對每個 cluster 獨立決定要叫誰的名字，結果**兩個 cluster 都選到 Alice** 的情況時有所聞（diarization 偶爾會把同一個人切成兩個 cluster）。

新版本：
1. 先把所有 (cluster, profile, 相似度) 收齊
2. 按相似度由高到低排序
3. 走過清單：cluster 還沒分配、enrolled name 還沒被搶 → 配對成功
4. 一個 enrolled name **最多只能被一個 cluster 領取**；後到的同名 cluster 退到 `speaker_N`

這樣 Alice 在輸出裡保證只會出現一次。

## 登錄聲紋是怎麼算出來的

放在 `<speakers>/Alice/` 下的三個音檔，**不是**黏在一起再抽聲紋，而是：

```
a.wav → 模型 → 聲紋 A → L2 normalize
b.wav → 模型 → 聲紋 B → L2 normalize  ⎫
c.wav → 模型 → 聲紋 C → L2 normalize  ⎬→ 算平均 → 再 L2 normalize
                                       ⎭                  ↑
                                                  這個就是 Alice 的代表聲紋
```

這是業界（WeSpeaker、ECAPA、Kaldi 等）一致的做法。**直接黏音檔再抽聲紋**會讓模型內部的統計層被不同麥克風/房間的特徵搞混，得到一個模糊的代表。

額外保護：**總語音時長 < 15 秒**會拒絕登錄並警告——太短的樣本算出來的聲紋不穩定，會污染後續匹配。

## 進階：Per-segment 再驗證（opt-in）

啟用 `--verify-segments` 之後，會多一步：

每個被標成 enrolled 名字（例如 Alice）的 ASR segment，**再抽一次該 segment 的聲紋**，跟 Alice 的聲紋比對。如果 cosine < `--verify-threshold`（預設 0.50），這一段標成 `Unknown`。

用途：cluster 整體是 Alice 沒錯，但裡面夾雜了 1 秒 Bob 講的話——舊版會把那 1 秒誤標成 Alice，開啟 verify 後會降級成 Unknown。

預設關閉，避免增加固定 30% 左右的計算量。研究調查中被點名為「最高 CP 值的改進」。

## 工具命令

| 命令 | 用途 |
|------|------|
| `metr speakers list` | 列出所有 enrolled 名字、樣本數、是否需要重新登錄 |
| `metr speakers inspect <name>` | **重要**：診斷某位 enrolled 人物的聲紋品質——印出每個音檔對代表聲紋的 cosine、跟其他人的最高相似度、安全 margin |
| `metr speakers verify --name X --audio f.wav` | 拿一段測試音檔比對 enrolled 人物 X |
| `metr <audio> --dry-run` | **快速調參**：只跑分群+認人，不跑 ASR/情緒/輸出。改 threshold/margin 立刻看結果 |
| `metr <audio> --no-discover` | 不要為未匹配的 cluster 自動建立 `speaker_N` 資料夾（cluster 仍會被標 `Unknown`，只是不入庫）。等同於在 `<speakers>/_metr/config.yaml` 設 `discover: false` |

## 調參建議

實務上對中文錄音建議從以下開始試：

| 場景 | match_threshold | match_margin | 備註 |
|------|----------------|--------------|------|
| 預設 | 0.65 | 0.07 | 一般會議 |
| 高品質錄音、想要少 false-positive | 0.70 | 0.10 | 寧可標 `speaker_N` 也不要叫錯名字 |
| 嘈雜環境、寧可有名字 | 0.60 | 0.05 | 接受偶爾誤認 |
| 中文（zh-TW / zh） | 0.68 | 0.07 | 因 cross-language cosine spread 較緊 |

修改後跑 `metr <audio> --dry-run --match-threshold 0.68` 看結果，再用
`metr speakers inspect <name>` 看各個 enrolled 人物的安全 margin。

## 限制

當前的 WeSpeaker 模型（FluidAudio 內建）是英文 VoxCeleb 訓練，**對中文有顯著
EER 退化**。要把中文表現拉到最好，需要切換到 CN-Celeb 訓練的 WeSpeaker
或 CAM++ 模型——這在 macOS 上目前還需要透過 sherpa-onnx 或 CoreML 自行
轉換，未列入這輪改動。

其他未做的：AS-Norm 分數正規化、TS-VAD 目標語者偵測，理由與工程量請見
CHANGELOG.md 的「Known follow-ups」。
