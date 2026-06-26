# 聲紋 Embedding 模型選型與授權決策筆記

> 調查日期：2026-06（7+ agent 兩輪查證）。**本筆記是事實彙整 + 工程判斷,非法律意見**;
> 商用上架前涉及授權/隱私的 go/no-go 仍需 IP/隱私律師確認(下方逐項標註)。

## 問題背景

目標:把 metr 包成可在 **Mac App Store 付費販售**的 on-device app。
現用聲紋模型 = FluidAudio 內建的 **WeSpeaker(VoxCeleb 訓練,256 維)**,觸發兩個疑慮:

1. **授權/來源**:VoxCeleb 音檔是 YouTube 爬來的。
2. **效能**:VoxCeleb 純英文訓練,套中文會議 EER 掉到十幾趴(見下「跨語言」)。

結論先講:**換掉 WeSpeaker 的首要理由其實是「中文效果差」,不是授權**;授權恐慌有一半是虛驚。

---

## 兩道授權閘(必須分開看)

| 閘 | 管什麼 | 法律性質 |
|----|--------|---------|
| **① 模型權重授權** | 你能不能散布/商用這個 `.onnx` 權重 | 模型作者的授權(Apache/MIT/custom) |
| **② 訓練資料授權** | 訓練資料的禁商用條款是否「傳染」到權重 | **法律未定論**(見下「法律」) |

模型權重標 Apache,**不等於**訓練資料乾淨;但訓練資料的限制是否真的下流到權重,無判例。

---

## 跨語言:訓練語言確實有影響

聲紋抓音色,但**不夠語言無關到可忽略**:

| 隔離「純語言」因素 | EER | 來源 |
|------|:---:|------|
| VoxCeleb1 → VoxCeleb1-B 雙語(同語料同模型,只換語言) | 1.17% → **5.96%(4-5×)** | arXiv:2211.00437 |
| VoxCeleb 訓練 → CN-Celeb 中文(未調適) | **17.78%** | arXiv:2206.07548 |

**關鍵轉折**:懲罰大,只發生在「訓練資料**未涵蓋**的語言」。多語/含中文訓練的 embedding 掉分小很多。
→ 正解不是「英文就行」,而是**用訓練含中文的 embedding**。現用 WeSpeaker(純英文)中文約十幾趴 EER。

**對 metr 用途的緩解**:做 nearest-of-N + 會議內聚類,絕對 EER 沒那麼要緊;唯一真痛點是
「拒絕陌生人」的 cosine 門檻(開集),中文不匹配會誤拒——**可用中文音檔重新校準門檻來救**。

---

## 候選模型最終判定(來源已釘死)

| 模型 | 權重授權 | 訓練資料(ModelScope 卡片原文) | VoxCeleb/CN-Celeb | 維度 | ONNX | 中文 | 判定 |
|------|:---:|------|:---:|:---:|:---:|:---:|:---:|
| **ERes2Net-large `_3dspeaker_16k`** ⭐ | Apache-2.0 | 僅 3D-Speaker 資料集(~10k 講者) | **都無** | 512 | ✅ | ✅ 原生 | ✅ **推薦** |
| ERes2Net-base `_3dspeaker_16k` | Apache-2.0 | 同上 | 都無 | 512 | ✅ | ✅ | ✅ |
| CAM++ `_3dspeaker_16k` | Apache-2.0 | 同上 | 都無 | 512 | ❌ 無 ONNX | ✅ | ✅(無 ONNX) |
| CAM++ `zh-cn-common`(200k) | Apache-2.0 | 「大型中文資料集」**未揭露明細** | ❓ 未揭露 | 192 | ✅ | ✅ 最強 | ⚠️ 無法證明乾淨 |
| ERes2NetV2 `zh-cn-common`(200k) | Apache-2.0 | 同上未揭露 | ❓ | 192 | ✅ | ✅ 最強 | ⚠️ 無法證明 |
| 現用 WeSpeaker(FluidAudio) | CC-BY | VoxCeleb | 有 | 256 | (CoreML) | ❌ 弱 | ⚠️ 中文差+同意層 |
| WeSpeaker/3D-Speaker CN-Celeb 系 | Apache | 含 **CN-Celeb(明文禁商用)** | 有 | — | ✅ | ✅ 強 | ❌ 違反 NC |
| VoxBlink2 / ECAPA2 / WSI | CC-BY-NC | — | — | — | — | — | ❌ NC 權重 |

**`_3dspeaker_16k` 系列的關鍵優勢**:訓練資料只有 **3D-Speaker 資料集**,而該資料集是
**錄音室多裝置實錄、非網路爬蟲**(arXiv:2306.15354,10k 講者/1124 小時/14 方言)。
這是少數能說「資料來源乾淨」的中文聲紋模型。

> ⚠️ 註:`iic/speech_eres2net_sv_zh-cn_3dspeaker_16k`(無 base/large)在 ModelScope **不存在**,
> 真實的是 `_base_` 與 `_large_` 兩個。

---

## SA vs NC 矛盾:已釐清,是虛驚

| 三道授權標示 | 實際授權 | 商用 |
|------|---------|:---:|
| **模型權重**(`_3dspeaker_16k`,5 變體逐一從 ModelScope API 確認) | **Apache-2.0** | ✅ |
| 訓練資料集(metadata) | CC-BY-SA-4.0 | ✅(標註+ShareAlike) |
| arXiv 論文 PDF | CC-BY-NC-SA | 與資料/權重無關 |

**NC 是論文 e-print 的授權**,arXiv 沒有機制去授權外部資料集。資料集本身是 CC-BY-SA(商用可);
最該看的權重直接 Apache-2.0。

**唯一殘留軟點**:官網只明寫 metadata 是 CC-BY-SA,**音檔本身授權未明講**。僅在「再散布資料集」時
咬到——metr 只用權重不散布資料集,實務不影響。要 airtight 可寄信問維護者確認音檔授權。**[需確認]**

---

## 業界實務(怎麼用、有沒有人商用)

**業界心智模型**:「模型卡標 Apache → 可商用出貨,訓練資料來源是作者的事」——**沒人往底下看**。

- **sherpa-onnx(重散布者)**:明確撇清——「每個模型有自己的授權,請自行查 repo」,不背書商用。
- **中文社群(知乎/CSDN/阿里雲)**:Apache = 商用可的清楚共識,對訓練資料來源零討論。
- **GitHub issues**:一片沉默,沒人問商用、無維護者裁示。
- **真實商用出貨薄**:確認的只有 **OpenWhispr**(2026/4,sherpa-onnx 端上跑 CAM++,自身 MIT);
  通義聽悟(傳聞未證實);主流 STT SaaS 不用 CAM++(走 pyannote/自研)。
- **對比**:同樣「Apache+NC 資料」爭議在 LLM 圈被罵「copyright laundering」(Normistral/Depth-Anything/
  F5-TTS),語者圈卻**沒人看**——共識是「沒人查過」的產物,非「查過放行」。

**對 metr 的意義**:業界 happily 出貨的是**更髒的** 200k-common(混 VoxCeleb+CN-Celeb);
你用 `_3dspeaker_16k` **兩道閘都過**,等於同時拿到「業界 common practice 的安全感」+「真正能講的來源故事」,
比多數人保守。

---

## 法律(2024-2026,資料→權重污染)

- **Getty v. Stability(英國高院 2025-11,最對題)**:不儲存原作的權重不算侵權物。聲紋只存統計向量 → 安全區。
- **美國著作權局(2025-05)**:污染以「是否記憶實質表達」為條件;192/512 維聲紋記不住。
- 打贏的案子命門都是**輸出替代**或**盜版取得資料**,聲紋兩者不沾。
- **真正殘留風險**(換批,且更該管):
  1. **資料集禁商用條款=合約**:避開 CN-Celeb/VoxBlink2(`_3dspeaker_16k` 已避開)。**[需律師]**
  2. **生物辨識隱私(BIPA/GDPR)**:聲紋是生物特徵,對消費級語音 app 可能是**最大實際曝險**,
     與選哪個模型無關。**[需律師]**

「權重是衍生作品 → app 侵權」是**最弱的理論**,對獨立付費 app 執法風險低。

---

## 推薦與後備

```
方案 A(推薦):ERes2Net-large _3dspeaker_16k
  · 權重 Apache-2.0、訓練資料乾淨可商用、中文原生、業界標準選擇
  · ONNX 現成 → 走「已整合的 sherpa-onnx」(不必碰 FluidAudio)
  · 工程量低;授權這關綠燈

方案 C(後備,若要完全零授權爭議):自訓
  · Common Voice zh(CC0) + AISHELL-1(Apache),~24k 講者
  · ~$200-600、幾天~兩週、完全自有
  · 中文原生、乾淨登錄場景 sub-2%

次選:
  方案 B:接受 VoxCeleb(律師清同意/隱私層)— 但中文差,不划算
  方案 D:買 Sensory TrulySecure(macOS+中文+端上,$2,500+權利金,要談上架權)
```

## 落地細節

1. **整合**:`_3dspeaker_16k` 只有 ONNX、**無 CoreML** → 走 sherpa-onnx(可能 CPU、非 ANE),
   比 FluidAudio 的 CoreML/ANE 略慢——這是「乾淨」的效能小代價。FluidAudio 內建 CoreML 反而是
   **無法證明乾淨的 200k-common CAM++**。
2. **維度變更**:256→512 → 既有聲紋全要重算。**已有 `enroll --rebuild`** 正為此設計,接得上。
3. **門檻校準**:用中文音檔重新校準拒絕陌生人的 cosine 門檻。

## 待辦(換模型前)

- [ ] **[需確認]** 寄信問 3D-Speaker 維護者:資料集**音檔**授權(CC-BY-SA 還是僅 metadata)
- [ ] **[需律師]** CN-Celeb 類 NC 條款已避開的確認 + 生物辨識隱私(BIPA/GDPR)合規
- [ ] 工程:sherpa-onnx 接 ERes2Net-large ONNX、`--rebuild` 重算、門檻校準

## 來源(關鍵)

- 權重授權:ModelScope API `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` → `"License":"Apache License 2.0"`
- 資料集授權:https://3dspeaker.github.io/ → "CC BY-SA 4.0"(metadata)
- 資料集實錄證明:arXiv:2306.15354(3D-Speaker dataset paper)
- 跨語言 EER:arXiv:2211.00437、arXiv:2206.07548
- 法律:Getty v. Stability [2025] EWHC 2863 (Ch);US Copyright Office "Copyright and AI" Part 3 (2025-05)
- 業界商用案例:https://openwhispr.com/blog/local-speaker-diarization
- sherpa-onnx 授權立場:https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
