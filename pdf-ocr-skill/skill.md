---
name: pdf-ocr
description: |
  PDF → JSONL，會先判斷是文字型還是掃描型：
  - 文字型：直接用 PyMuPDF 抽文字（不浪費 token）
  - 掃描型：用 LLM 視覺能力（Read / view_image）逐頁辨識
  嚴禁任何傳統 OCR 引擎（Tesseract / PaddleOCR / EasyOCR）、雲端 OCR API、MCP 圖像工具。
  一次只做一本書（單機鎖），完成後清除所有暫存（PNG 與 progress 檔）。
  每次最多處理一個 batch（預設 30 頁），支援 --resume。
  觸發時機：
  - 使用者說「OCR 這個 PDF」
  - 使用者說「把掃描 PDF 做成 JSONL」
  - 使用者說「/pdf-ocr <path>」
  - 使用者說「用 LLM 視覺辨識這個 PDF」
  典型呼叫：
  - /pdf-ocr /path/to/book.pdf
  - /pdf-ocr /path/to/book.pdf /path/to/output.jsonl
  - /pdf-ocr /path/to/book.pdf --resume
  - /pdf-ocr /path/to/book.pdf --batch 20 --dpi 120
---

# PDF → JSONL（智慧路由：文字型直取 / 掃描型 LLM Vision）

> **核心原則**：本技能先判斷 PDF 類型再決定做法。
> - 有文字層的 PDF → **直接抽文字**（PyMuPDF），零 token 成本
> - 掃描型 PDF → **LLM 視覺辨識**（Read / view_image），唯一允許的辨識方式
> - 嚴禁任何傳統 OCR 引擎、雲端 OCR API、第三方 OCR 服務、MCP 圖像工具
>
> 這些規則不可協商，違反即視為任務失敗。

---

## 強制規則（Hard Rules — 違反即任務失敗）

### A. 永不允許的辨識方式

| 類別 | 禁止項目（範例） |
|---|---|
| 命令列 OCR 引擎 | `tesseract`, `paddleocr`, `easyocr`, `cnocr`, `ocrmypdf` |
| Python OCR 套件 | `pytesseract`, `paddlepaddle-ocr`, `easyocr`, `mmocr`, `cnocr` |
| 雲端 OCR API | Google Cloud Vision OCR、Azure Computer Vision、AWS Textract、阿里雲 / 騰訊雲 / 百度 / 有道 OCR |
| 第三方 OCR 服務 | ABBYY FineReader、Adobe Acrobat OCR、商務 OCR SaaS |
| MCP 圖像理解工具 | `mcp__MiniMax__understand_image`、任何 `mcp__*` 圖像相關工具 |
| 自寫包裝 | 自行寫的 Tesseract / Paddle / EasyOCR wrapper、subprocess 包裝、shell 腳本呼叫 OCR 引擎 |

> 若你正準備執行 `tesseract`、`paddleocr`、`easyocr`、`pytesseract` 等指令，**立刻停止並告知使用者你正準備違反 skill 規則**，請使用者裁示是否取消任務。

### B. 兩條合法路徑

| 路徑 | 適用 | 工具 | source 欄位 |
|---|---|---|---|
| 文字直取 | 有文字層的 PDF（電子書、有 OCR 過的 PDF） | PyMuPDF `page.get_text()` | `pdf_text_layer` |
| LLM Vision | 掃描型 PDF（純圖像、古籍、影印本） | Claude Read / Codex view_image | `llm_vision` |

「LLM Vision 辨識」 = 讓 LLM 本身直接看 PNG 圖片。**禁止**先用 OCR 引擎轉文字再交給 LLM 整理。

### C. JSONL 必填驗證欄位

每一筆 record 必須包含：

```json
{
  "loc": {"item_index": 0, "chunk_index": 0},
  "chapter": "第一章：命宮",
  "text": "...",
  "source": "llm_vision"
}
```

`source` 欄位**必填**，且僅能為以下兩值之一：
- `"llm_vision"`（掃描型 PDF 走 LLM Vision 辨識）
- `"pdf_text_layer"`（文字型 PDF 走 PyMuPDF 抽文字）

任何其他值（`tesseract`, `paddleocr`, `chi_tra_vert`, `mcp_minimax`, `manual` 等）即視為違規輸出，**整批作廢重做**，不可只修補單筆。

### D. 既有 JSONL 完整性檢查（Step 0 強制執行）

執行 `--resume` 或對既有 JSONL 重做時，必須先：

1. 讀取 JSONL 第一筆 record，檢查 `source` 欄位
2. 若 `source` 不在合法清單（`llm_vision` / `pdf_text_layer`）：
   - 標示此 JSONL 為「已污染」
   - 立即通知使用者來自非法 OCR 流程
   - **不可使用 `--resume` 沿用**，只能整本重做
   - 建議刪除該檔再啟動全新 OCR
3. 若 `source` 欄位不存在：
   - 視為可疑，告知使用者，請求確認

### E. Bash 指令防火牆

執行流程中，禁止透過 Bash 工具呼叫下列關鍵字：

- `tesseract`, `pytesseract`
- `paddleocr`, `paddle_ocr`, `paddleocr-cli`, `paddle ocr`
- `easyocr`, `cnocr`, `ocrmypdf`
- 雲端 OCR CLI：`gcloud vision`, `aws textract`, `az cognitiveservices vision`, `aliyun ocr`, `tencent ocr`

如使用者明確要求改用上述工具，禮貌拒絕並說明本 skill 唯一支援文字直取 / LLM Vision；使用者堅持，請使用者離開本 skill 自行處理。

### F. 一次只做一本書（單機鎖機制）

啟動前必須確認**目前沒有其他 OCR 任務進行中**：

1. 掃描 `/tmp/pdf-ocr-*/` 與 `*.ocr-progress.json` 是否存在
2. 若有 ≥ 1 個未完成的進度檔（`completed_pages 數 < total_pages`）：
   - **拒絕啟動新書 OCR**
   - 告知使用者：「目前還有 `<那本.pdf>` 在進行中（X/Y 頁），請先完成或清除」
   - 請使用者選擇：
     - (a) 先 `--resume` 完成那本
     - (b) 手動刪除 `*.ocr-progress.json` 與對應 `/tmp/pdf-ocr-xxx/` 後再開新書

> 平行多本會塞爆暫存空間、混淆進度、每批接近 context 上限只會降低品質。**強制串行。**

### G. 完成後清除暫存（強制執行）

全部頁面 OCR 完成（Step 6 寫出最終 JSONL）後，必須：

1. 刪除 PROGRESS_FILE：`{OUTPUT_PATH}.ocr-progress.json`
2. 刪除 PNG 暫存目錄：`/tmp/pdf-ocr-xxxxxxxx/`（含其下所有 `page_*.png`）
3. 確認 `/tmp/pdf-ocr-*/` 不再有此次任務殘留
4. 回報「✅ 暫存已清除」

> 使用者只應拿到輸出的 JSONL 檔案，不該看到 progress 或 PNG 殘留。
> 文字直取路徑沒有 PNG 暫存，仍須確認沒生出多餘檔案。

---

## 參數解析

| 參數 | 說明 | 預設值 |
|---|---|---|
| `PDF_PATH` | PDF 檔案路徑（必要） | — |
| `OUTPUT_PATH` | 輸出 JSONL 路徑（選用） | PDF 同目錄 + .jsonl |
| `--resume` | 從上次中斷點繼續 | false |
| `--batch N` | 每次最多處理頁數（僅 LLM Vision 路徑） | 30 |
| `--dpi N` | 圖片渲染解析度（僅 LLM Vision 路徑） | 120 |
| `--force-ocr` | 強制走 LLM Vision，跳過文字型偵測 | false |

**PROGRESS_FILE** = `OUTPUT_PATH + ".ocr-progress.json"`

---

## Step 0：前置檢查

1. 確認 `PDF_PATH` 存在
2. 若 `OUTPUT_PATH` 未指定，設為 `PDF_PATH` 同目錄、副檔名換成 jsonl
3. **§F 單機鎖檢查**：掃描 `/tmp/pdf-ocr-*` 與既有 `*.ocr-progress.json`，若有未完成任務 → 依 §F 處理
4. **§D 既有 JSONL 完整性驗證**：若 `OUTPUT_PATH` 已存在，讀第一筆檢查 `source` 欄位

---

## Step 0.5：判斷 PDF 類型（強制執行 — 避免浪費 token）

在啟動任何 PNG 渲染或 LLM Vision 前，必須先判斷 PDF 是文字型還是掃描型：

```bash
# Claude Code：
python3 ~/.claude/skills/pdf-ocr/detect_pdf_type.py "<PDF_PATH>"

# Codex：
python3 ~/.codex/skills/pdf-ocr/detect_pdf_type.py "<PDF_PATH>"
```

讀取輸出的 `TYPE:` 與 `AVG_CHARS_PER_PAGE:`：

| TYPE | 平均字元/頁 | 處理方式 |
|---|---|---|
| `text` | ≥ 50 | **走 Step 1A 文字直取路徑**（不啟動 LLM Vision） |
| `scanned` | < 50 | 走 Step 1B–6 LLM Vision 路徑 |
| 邊界 | 30–50 | 抽樣讀 1–2 頁 PNG 確認後再決定 |

如使用者用 `--force-ocr` 旗標強制走 OCR，跳過此判斷直接進 Step 1B。

> **過往教訓**：曾把文字型電子書（裡面就有可抽的文字）硬走 OCR，浪費大量 token。**判斷錯了等於 skill 失敗。**

---

## Step 1A：文字直取路徑（適用 `TYPE:text`）

直接用 PyMuPDF 抽文字、切 chunks、寫 JSONL，**完全不渲染 PNG、不消耗 vision token**：

```python
import fitz, json, re
doc = fitz.open(PDF_PATH)
records = []
current_chapter = "（前言）"
current_item_index = 0
chunk_counts = {}

CHAPTER_RE = re.compile(r'(第[一二三四五六七八九十百千零〇\d]+[章節篇回卷]|序|前言|目[錄录]|附[錄录])')

for page_idx, page in enumerate(doc):
    text = page.get_text().strip()
    if not text:
        continue
    # 簡易章節偵測：第一行若像章節標題就更新 current_chapter
    first_line = text.split('\n', 1)[0].strip()
    if CHAPTER_RE.search(first_line) and len(first_line) <= 30:
        current_chapter = first_line
        current_item_index += 1
    # 切 chunks（≤ 500 字，優先在段落或句號邊界）
    chunks = chunk_text(text, max_chars=500)
    ck = str(current_item_index)
    chunk_counts.setdefault(ck, 0)
    for chunk in chunks:
        records.append({
            "loc": {"item_index": current_item_index, "chunk_index": chunk_counts[ck]},
            "chapter": current_chapter,
            "text": chunk,
            "source": "pdf_text_layer"
        })
        chunk_counts[ck] += 1

doc.close()

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
```

完成後：
- 文字直取**不需要** PROGRESS_FILE、不需要 PNG 暫存（這條路徑沒有暫存）
- 直接跳到 §G 清理（其實沒東西可清，但仍要回報）
- 跳到 Step 6 結束

---

## Step 1B：提取頁面圖片（LLM Vision 路徑專用）

執行 extract_pages.py 把 PDF 轉成 PNG 圖片：

```bash
python3 ~/.claude/skills/pdf-ocr/extract_pages.py \
  "<PDF_PATH>" \
  "<PNG_DIR>" \
  --dpi <DPI>
```

- `PNG_DIR` = 若進度中已有 `png_dir` 且目錄存在則沿用，否則用 `/tmp/pdf-ocr-<uuid4前8碼>/`
- 腳本輸出最後三行包含 `TOTAL:<n>`、`EXTRACTED:<n>`、`OUT_DIR:<path>`

> extract_pages.py 只負責 PDF→PNG 渲染（PyMuPDF 純圖像轉換），**不做任何 OCR**。

非 `--resume` 時初始化進度結構（見 §進度結構）。`--resume` 則讀取既有 `PROGRESS_FILE`。

---

## Step 2：決定本次 batch 範圍

```
已完成 = PROGRESS.completed_pages（整數 list，0-based）
待處理 = sorted([i for i in range(TOTAL_PAGES) if i not in 已完成])
本次 batch = 待處理[:BATCH_SIZE]
```

若 `本次 batch` 為空 → 跳到 Step 5

告知使用者：「本次處理第 {min+1}–{max+1} 頁（共 {len(batch)} 頁）...」

---

## Step 3：逐頁 LLM Vision OCR

**嚴格遵守 §A、§B 規則。** 每頁辨識都必須用 §B「LLM Vision」路徑工具。

### 3a. 用平台讀圖工具讀取圖片

- Claude Code：`Read("/tmp/pdf-ocr-xxx/page_{page_idx:04d}.png")`
- Codex：`view_image` 讀取同樣路徑

模型視覺能力會自動辨識頁面內容。

### 3b. 判斷頁面類型並轉錄

| 類型 | 判斷標準 | 處理方式 |
|---|---|---|
| 空白 / 封面 / 純裝飾圖 | 幾乎無文字、無知識性內容 | SKIP，記錄 `skip=true` |
| 章節標題頁 | 大標題（居中、字體大、第X章 等） | 更新 `current_chapter`，`item_index +1` |
| 一般文字頁 | 連續文字段落 | 轉錄原文 |
| 有意義的圖示頁 | 流程圖、表格、命盤、教學插圖 | 用文字描述圖示傳達的知識重點 |

**圖示頁格式**：
```
【圖示說明】<一句話說明圖的類型>
<描述圖中的主要內容、結構、概念重點，2–5 句話>
<關鍵數字、名詞、步驟順序明確列出>
```

**轉錄原則**

- 完整逐字轉錄，保留標點符號
- **直書（垂直書寫）書籍**：模型自動依語意辨識欄位順序，不論古籍由右到左、現代版由左到右皆能正確處理；不必特別調參
- 一頁同時有標題和正文，先記錄章節名再轉錄正文
- 頁碼、書名行等重複性 header/footer 可省略
- 圖示描述雖非原文，仍建立為正式 record

**裝飾性，應 SKIP**：純底色、章節分隔頁、封面封底、版權頁、作者照片風景照。

### 3c. 切 chunks

≤ 500 字，優先在段落 `\n` 或句號 `。` 邊界切割。

### 3d. 建立 records

```json
{
  "loc": {"item_index": <current_item_index>, "chunk_index": <ci>},
  "chapter": "<current_chapter>",
  "text": "<chunk_text>",
  "source": "llm_vision"
}
```

**`source` 必填且必為 `"llm_vision"`**（§C 規則）。

### 3e. 更新進度並立即存檔

更新 `PROGRESS`：
- `completed_pages` 加入 `page_idx`
- `current_chapter`、`current_item_index`、`item_chunk_counts` 更新
- `records` 加入新 records

立即寫入 `PROGRESS_FILE`（每頁完成都要存）。

---

## Step 4：回報進度

每完成 5 頁，輸出一行：
```
[15/30] 頁 16-20：已完成，累計 87 records，目前章節：第三章：命宮
```

---

## Step 5：寫出 JSONL + 健全性檢查

從 `PROGRESS.records` 寫出：

```python
import json
records = json.load(open(PROGRESS_FILE))["records"]
# §C 規則：每筆 source 必須是合法值
LEGAL = {"llm_vision", "pdf_text_layer"}
for r in records:
    assert r.get("source") in LEGAL, f"違規 record：{r}"
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
```

assert 失敗即 §C 違規，整批作廢重做。

### 5b. 輸出健全性快檢

寫出後讀回前 3 筆與隨機 3 筆，目視檢查：
- 是否為連貫中文，而非「字 字 字」這類 OCR 引擎特徵亂碼
- 是否有「亂入」英文字母（OCR 引擎常把直書「川」誤判成 "lll"）
- 中文標點是否完整保留

發現亂碼特徵 → 立刻停下並告知使用者疑似違規。

---

## Step 6：結束 + 清除暫存（§G 強制執行）

**若還有剩餘頁面**：
```
✅ 本次完成 {len(batch)} 頁（第 {start+1}–{end+1} 頁）
📄 累計 {total_records} records，JSONL 已更新：{OUTPUT_PATH}
📌 剩餘 {remaining} 頁，請執行：
   /pdf-ocr {PDF_PATH} --resume
```

**若全部完成（§G 清理）**：

```bash
# 1. 刪 PROGRESS_FILE
rm -f "{OUTPUT_PATH}.ocr-progress.json"

# 2. 刪 PNG 暫存目錄（僅 LLM Vision 路徑有此目錄）
rm -rf "{PNG_DIR}"

# 3. 確認沒殘留
ls /tmp/pdf-ocr-* 2>/dev/null  # 應該不再有此次任務的目錄
```

回報：
```
🎉 完成！
📄 共 {total_pages} 頁，{total_records} records
🔧 路徑：{TYPE：text 直取 / 掃描型 LLM Vision}
💾 輸出：{OUTPUT_PATH}
✅ 來源驗證：所有 record source ∈ {llm_vision, pdf_text_layer}
🧹 暫存已清除：PROGRESS_FILE 與 PNG 目錄已刪
➡️  下一步：「把 {OUTPUT_PATH} 做成 skill」
```

---

## 進度結構（PROGRESS_FILE 格式）

僅 LLM Vision 路徑使用：

```json
{
  "pdf_path": "/path/to/book.pdf",
  "output_path": "/path/to/output.jsonl",
  "total_pages": 419,
  "png_dir": "/tmp/pdf-ocr-a1b2c3d4",
  "completed_pages": [0, 1, 2, 3],
  "current_chapter": "第一章：命宮",
  "current_item_index": 2,
  "item_chunk_counts": {"0": 3, "1": 2, "2": 1},
  "records": [
    {"loc": {"item_index": 0, "chunk_index": 0}, "chapter": "（前言）", "text": "...", "source": "llm_vision"}
  ]
}
```

文字直取路徑一次跑完，不需要 progress 檔。

---

## 誠信原則（不可違反）

> 這些原則源自三次真實的嚴重錯誤：
> 1. 聲稱完成 OCR，但 JSONL 內容是自行捏造的摘要，非真實提取。
> 2. 用 Tesseract OCR（`source: tesseract_chi_tra_vert`）跑了「劝学斋紫微初阶」，輸出滿篇亂碼後仍交差。
> 3. 把文字型電子書硬走 LLM Vision OCR，浪費大量 token，未先判斷 PDF 類型。

1. **禁止偽造輸出**：JSONL 每一筆 `text` 必須是真實提取（文字直取）或模型視覺辨識（LLM Vision），不可摘要、推測、自行撰寫。
2. **必須走合法路徑**：用戶指定 `/pdf-ocr` 時，依 §B 兩條合法路徑之一執行；禁止為了求快或求省 token 而改用 §A 禁止項目。
3. **必須先判斷類型**：未跑 detect_pdf_type.py 不可進入 Step 1B。文字型 PDF 走 OCR 即視為違規（除非 `--force-ocr` 明確指示）。
4. **一次一本書**：違反 §F 平行處理多本即視為違規。
5. **完成必清乾淨**：違反 §G 留暫存即視為違規。
6. **聲稱完成前必須驗證**：（a）內容真實、（b）records 數量符合頁數、（c）每筆 source 合法、（d）暫存已清。
7. **無法完成時誠實說明**：能力不足立即告知，不可偽裝成功，更不可改用 §A 禁止項目硬幹。

---

## 注意事項

- **文字直取路徑**：不耗 vision token，速度快，適用任何有可選取文字的 PDF
- **LLM Vision 模式**：Read tool / view_image 讀取 PNG，模型視覺自動辨識，無需 API key
- 每次 LLM Vision batch 最多 30 頁（預設），超過會因 context 過大影響品質
- DPI 120 適合一般中文書籍；手寫、模糊或直書古籍建議 `--dpi 150`
- 若某頁讀圖失敗，記錄 `skip=true` 並繼續
