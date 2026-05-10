---
name: pdf-ocr
description: |
  PDF → JSONL，會先判斷是文字型還是掃描型：
  - 文字型：直接用 PyMuPDF 抽文字（不浪費 token）
  - 掃描型：用 LLM 視覺能力（Read / view_image）逐頁辨識
  嚴禁任何傳統 OCR 引擎（Tesseract / PaddleOCR / EasyOCR）、雲端 OCR API、MCP 圖像工具。
  一次只做一本書（單機鎖），完成後清除所有暫存（PNG 與 progress 檔）。
  每次最多處理一個 batch（預設 8 頁），支援 --resume；長時間任務必須採用低上下文 append-only 流程。
  觸發時機：
  - 使用者說「OCR 這個 PDF」
  - 使用者說「把掃描 PDF 做成 JSONL」
  - 使用者說「/pdf-ocr <path>」
  - 使用者說「用 LLM 視覺辨識這個 PDF」
  典型呼叫：
  - /pdf-ocr /path/to/book.pdf
  - /pdf-ocr /path/to/book.pdf /path/to/output.jsonl
  - /pdf-ocr /path/to/book.pdf --resume
  - /pdf-ocr /path/to/book.pdf --batch 8 --dpi 120
hooks:
  TaskCreated:
    - hooks:
        - type: command
          command: 'python3 "${CLAUDE_SKILL_DIR}/pdf_ocr_guard.py" check-subagent-hook'
          statusMessage: "Checking pdf-ocr worker isolation..."
  SubagentStart:
    - hooks:
        - type: command
          command: 'python3 "${CLAUDE_SKILL_DIR}/pdf_ocr_guard.py" check-subagent-hook'
          statusMessage: "Checking pdf-ocr worker isolation..."
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

### 0. Guard Hook 機制（強制使用）

本 skill 內建 `pdf_ocr_guard.py` 作為可執行的 guard hook。Codex/Claude 目前不會自動在每個 shell tool call 前注入 skill hook，因此使用本 skill 時必須在下列檢查點**手動執行 guard**；少跑任一檢查即視為流程不完整：

| 檢查點 | 必跑指令 | 目的 |
|---|---|---|
| 啟動前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-single-book --current-output "<OUTPUT_PATH>"` | 阻止同時處理多本書 |
| 啟動前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-duplicate-output "<PDF_PATH>" "<OUTPUT_PATH>"` | 阻止同一本 PDF 已有同名 JSONL 卻在 `ocr-jsonl/` 另做一份 |
| 啟動/續跑前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-existing-jsonl "<OUTPUT_PATH>"` | 阻止沿用污染 JSONL |
| 續跑前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py validate-progress "<PROGRESS_FILE>"` | 阻止沿用高上下文舊 progress |
| 每次準備執行 shell 命令前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-command -- <COMMAND...>` | 阻止 Tesseract/Paddle/EasyOCR/雲端 OCR/MCP 圖像工具 |
| 建立 worker subagent 時 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-subagent-hook` | 阻止非 fresh worker、fork 主上下文或缺少任務包的 OCR subagent |
| 渲染後 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py write-manifest "<PDF_PATH>" "<OUTPUT_PATH>" --png-dir "<PNG_DIR>"` | 保存 PDF/頁圖 SHA256 證據 |
| 寫出 JSONL 後 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py validate-jsonl "<OUTPUT_PATH>"` | 驗證 source、頁序、record 序 |
| 完成前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py finalize-audit "<OUTPUT_PATH>" --total-pages <TOTAL>` | 產生 final hash chain |
| 刪 PDF 前 | `python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py deletion-gate "<OUTPUT_PATH>" --total-pages <TOTAL> --keep-pages` | 確認可刪 PDF 的條件 |

Claude Code 安裝位置若是 `~/.claude/skills/pdf-ocr/`，同一套指令改用該路徑。

Claude Code 會透過本 skill frontmatter 的 `TaskCreated` / `SubagentStart` hook 自動執行 `check-subagent-hook`。Codex CLI 0.128 的 lifecycle hooks 目前不攔截 `spawn_agent` 建立事件；Codex 執行本 skill 時仍必須使用 `spawn_agent(..., fork_context:false, agent_type:"worker")`，並可用 `check-subagent-payload` 驗證任務包。

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
  "record_index": 0,
  "page_index": 0,
  "page_number": 1,
  "page_record_index": 0,
  "loc": {"item_index": 0, "chunk_index": 0},
  "chapter": "第一章：命宮",
  "text": "...",
  "source": "llm_vision",
  "page_sha256": "<該頁 PNG 的 SHA256>"
}
```

`source` 欄位**必填**，且僅能為以下兩值之一：
- `"llm_vision"`（掃描型 PDF 走 LLM Vision 辨識）
- `"pdf_text_layer"`（文字型 PDF 走 PyMuPDF 抽文字）

任何其他值（`tesseract`, `paddleocr`, `chi_tra_vert`, `mcp_minimax`, `manual` 等）即視為違規輸出，**整批作廢重做**，不可只修補單筆。

### C2. 頁序與忠實性欄位（Hard Audit）

為避免 append 亂序、漏頁、重複頁、上下文壓縮失憶或 worker 偷吃步，每筆 JSONL 必須滿足：

| 欄位 | 規則 |
|---|---|
| `record_index` | 全檔從 0 開始，逐筆 +1，不可跳號、重複或倒退 |
| `page_index` | 0-based PDF 頁碼，必須單調不倒退 |
| `page_number` | 必須等於 `page_index + 1` |
| `page_record_index` | 同一頁內從 0 開始逐筆 +1 |
| `page_sha256` | LLM Vision 路徑必填，等於該頁 PNG 在 manifest 中的 SHA256 |
| `text` | 必須是逐字轉錄或圖示忠實描述；不可摘要、補寫、推測 |

每頁即使 SKIP，也要寫一筆 `skip:true` record，包含 `skip_reason`、`page_index`、`page_number`、`page_sha256`。這樣 `validate-coverage` 才能證明整本每一頁都被看過。

### D. 既有 JSONL 完整性檢查（Step 0 強制執行）

#### D0. 目錄批次處理的既有成果優先規則

處理整個資料夾時，必須先掃描每個 PDF 同目錄是否已有同名 JSONL：

- 對 `/path/book.pdf`，若 `/path/book.jsonl` 已存在，預設視為已完成成果，必須跳過該 PDF。
- 不可因為要把輸出集中到 `ocr-jsonl/`，就另建 `/path/ocr-jsonl/.../book.pdf.jsonl` 重做同一本書。
- 若既有 JSONL 的 schema 不同於 pdf-ocr 新 schema，仍不可直接重做；必須回報使用者「已有舊成果，需要轉換/標準化或明確重做」。
- 只有使用者明確指定「重做、覆蓋、忽略原 JSONL」時，才能用新輸出路徑；並且要先說明會消耗 token。

啟動 OCR 前必跑：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-duplicate-output \
  "<PDF_PATH>" "<OUTPUT_PATH>"
```

若要明確重做，必須使用：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-duplicate-output \
  "<PDF_PATH>" "<OUTPUT_PATH>" --force-duplicate-output
```

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

每次準備執行 shell 命令前，先用 guard 檢查完整命令字串：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-command -- \
  python3 ~/.codex/skills/pdf-ocr/detect_pdf_type.py "<PDF_PATH>"
```

若 guard 輸出 `BLOCKED:` 或非 0 exit code，必須停止，不可改寫命令繞過。

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

全部頁面 OCR 完成（Step 6 寫出最終 JSONL）後，必須先完成 `finalize-audit`。PNG 是否刪除取決於 PDF 保存策略：

1. 刪除 PROGRESS_FILE：`{OUTPUT_PATH}.ocr-progress.json`
2. 若保留原始 PDF，才可刪除 PNG 暫存目錄：`/tmp/pdf-ocr-xxxxxxxx/`
3. 若準備刪 PDF，必須先把 PNG 移到正式 archive/workdir，不可刪 PNG
4. 確認本次任務沒有未審計暫存
5. 回報「✅ 暫存/證據保存策略已完成」

> 使用者只應拿到輸出的 JSONL 檔案，不該看到 progress 或 PNG 殘留。
> 文字直取路徑沒有 PNG 暫存，仍須確認沒生出多餘檔案。

### H. 長時間上下文控制（10 小時任務強制）

掃描型 LLM Vision OCR 可能跑數小時到十多小時。為避免主上下文滿出、壓縮後遺失狀態，必須遵守：

1. **不可把整本 records 放進對話或 progress**：`PROGRESS_FILE` 只保存狀態，不保存全文 records。
2. **JSONL 採 append-only**：每完成一頁，立即把該頁 records 追加到 `OUTPUT_PATH`，再更新 `completed_pages`。
3. **主上下文只保留控制資訊**：PDF 路徑、輸出路徑、總頁數、已完成頁數、目前章節、下一批頁碼、最近 1-3 筆短摘要。不要貼整頁 OCR 文字回主對話。
4. **預設 batch 降為 8 頁**：一般書籍用 `--batch 8`；密集古籍、直書、表格多時用 `--batch 3` 到 `--batch 5`。
5. **每 60-90 分鐘做一次 checkpoint**：執行 `validate-jsonl`，回報完成頁數與剩餘頁數即可，不貼內容。
6. **上下文壓縮前後可恢復**：恢復工作只依賴 `OUTPUT_PATH`、`PROGRESS_FILE`、`PNG_DIR`，不可依賴對話裡的未落盤內容。
7. **每個 batch 前讀取最後狀態**：worker 必須從 `PROGRESS_FILE` 與 `OUTPUT_PATH` 最後一筆取得 `record_index`、`current_chapter`、`item_chunk_counts`，不可憑對話記憶續接。
8. **每個 batch 後做硬驗證**：必跑 `validate-jsonl`；失敗即停止，不可繼續 append。
9. **正式 artifact 不可覆寫**：page artifact、batch manifest、run manifest、final report、deletion report 都用 run id/generation id；重試時產生新 generation，不覆寫舊正式檔。
10. **原子寫入**：所有正式 JSON artifact 必須先寫 `*.tmp`，flush/fsync 後 atomic rename，再 fsync parent directory。

#### Subagent / Fresh Worker 模式（建議用於長時間 OCR）

若平台支援 subagent，掃描型 OCR 應使用「每 batch 一個 fresh worker」：

- 主 agent 只負責：前置檢查、渲染 PNG、分配頁碼、驗證 JSONL、清理暫存。
- worker agent 負責：讀取本批 PNG、用 LLM Vision 轉錄、append JSONL、更新 progress。
- worker 必須是**全新上下文**：不要 fork 主對話全文；只傳必要任務包。
- 每個 worker 完成後關閉；下一批重新開新 worker，避免同一個 worker 累積十小時上下文。
- 不要平行開多個 worker 處理同一本書。單書仍然串行，避免章節狀態與 chunk index 競爭。

worker 任務包只包含：

```text
使用 pdf-ocr skill 的 LLM Vision 路徑。禁止 Tesseract/Paddle/EasyOCR/雲端 OCR/MCP 圖像工具。
PDF_PATH=<...>
OUTPUT_PATH=<...>
PROGRESS_FILE=<...>
PNG_DIR=<...>
PAGE_RANGE=<start-end, 1-based>
CURRENT_CHAPTER=<...>
CURRENT_ITEM_INDEX=<...>
ITEM_CHUNK_COUNTS=<...>
NEXT_RECORD_INDEX=<從 OUTPUT_PATH 最後一筆 + 1 得出>
請逐頁 view_image/Read PNG，轉錄後 append JSONL；每頁完成立刻更新 progress。
回覆只給：完成頁數、records 數、最後章節、是否有異常。不要貼全文。
```

Codex 可用 `spawn_agent`，設定 `fork_context:false`，agent type 用 `worker`；Claude Code 若沒有等效能力，則用短 batch + `--resume` 重啟新會話達到同樣目的。

建立 worker 前若平台沒有自動 Task/Subagent hook，先把即將送出的任務包寫成 JSON 後執行：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-subagent-payload "<TASK_PAYLOAD_JSON>"
```

### I. 原始資料保存與刪除限制

**在 JSONL 完成後立刻刪 PDF 是高風險操作，預設禁止。** 因為 JSONL 是衍生資料，不等同原始掃描證據。若真的需要刪除 PDF，必須先完成以下全部條件：

1. 產生 `{OUTPUT_PATH}.manifest.json`，包含原始 PDF SHA256、每頁 PNG SHA256、輸出 JSONL 路徑、`total_pages`。
2. 完成 `finalize-audit`，產生 `{OUTPUT_PATH}.final-report.json`。
3. 至少保留以下二選一：
   - 原始 PDF；或
   - 每頁 PNG + manifest + JSONL。
4. 執行 `deletion-gate`；若 PDF 與 PNG 都要刪除，guard 必須拒絕，且不可宣稱可審計。

最佳實務：把原始 PDF 移到 `archive/` 或冷儲存，不要刪除；若要節省空間，優先刪 `/tmp/pdf-ocr-*` PNG，保留 PDF + JSONL + manifest。

### J. 95 分模式：Artifact / Hash Chain / Lease

若目標是 24 小時長跑後允許刪 PDF，必須啟用 95 分模式：

1. **正式 workdir**：使用 `<OUTPUT>.ocr-work/`，不要把唯一頁圖證據放 `/tmp`。
2. **page artifact**：worker 不直接 append 最終 JSONL；每頁先產生 `pages_ocr/page_000001.g0001.ocr.json`。
3. **batch manifest**：每 batch 產生 `batch_manifest.json`，列出 page artifact SHA256、worker id、lease token、generation。
4. **central state**：主 agent 維護 run state，worker 只提交 completion manifest，不直接宣告全域完成。
5. **lease token**：每個 batch 有 lease token、created_at、expires_at；逾期後只能用新 generation 重試。
6. **canonical assembler**：最終 JSONL 只能由 deterministic assembler 依 `page_index ASC, page_record_index ASC` 組裝，`record_index` 由 assembler 產生。
7. **hash chain**：page artifact hash -> batch manifest hash -> run manifest hash -> final report hash -> deletion report hash。
8. **QA layer**：文字正確性不能由 hash 證明；必須標記低信心頁、抽樣人工審查或第二 pass 差異審查。

---

## 參數解析

| 參數 | 說明 | 預設值 |
|---|---|---|
| `PDF_PATH` | PDF 檔案路徑（必要） | — |
| `OUTPUT_PATH` | 輸出 JSONL 路徑（選用） | PDF 同目錄 + .jsonl |
| `--resume` | 從上次中斷點繼續 | false |
| `--batch N` | 每次最多處理頁數（僅 LLM Vision 路徑） | 8 |
| `--dpi N` | 圖片渲染解析度（僅 LLM Vision 路徑） | 120 |
| `--force-ocr` | 強制走 LLM Vision，跳過文字型偵測 | false |
| `--long-run` | 啟用長時間低上下文模式：batch 預設 8、append-only、可用 fresh worker | true |

**PROGRESS_FILE** = `OUTPUT_PATH + ".ocr-progress.json"`

---

## Step 0：前置檢查

1. 確認 `PDF_PATH` 存在
2. 若 `OUTPUT_PATH` 未指定，設為 `PDF_PATH` 同目錄、副檔名換成 jsonl
3. **Guard：單機鎖檢查**：
   ```bash
   python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-single-book --current-output "<OUTPUT_PATH>"
   ```
   若有未完成任務 → 依 §F 處理
4. **Guard：同名既有成果檢查**：
   ```bash
   python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-duplicate-output "<PDF_PATH>" "<OUTPUT_PATH>"
   ```
   若 PDF 同目錄已有 `<PDF_STEM>.jsonl`，必須跳過或請使用者明確確認重做，不可另輸出到 `ocr-jsonl/` 製造重複成果。
5. **Guard：既有 JSONL 完整性驗證**：
   ```bash
   python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-existing-jsonl "<OUTPUT_PATH>"
   ```
   若 `OUTPUT_PATH` 已存在且 `source` 不合法，視為污染，不可 `--resume`
6. **Guard：progress 狀態檔驗證**：
   ```bash
   python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py validate-progress "<OUTPUT_PATH>.ocr-progress.json"
   ```
   若 progress 內含完整 `records`，代表是舊版高上下文格式；必須先遷移或重建 progress，不可直接續跑

---

## Step 0.5：判斷 PDF 類型（強制執行 — 避免浪費 token）

在啟動任何 PNG 渲染或 LLM Vision 前，必須先判斷 PDF 是文字型還是掃描型：

```bash
# 先做 shell 命令 guard
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-command -- \
  python3 ~/.codex/skills/pdf-ocr/detect_pdf_type.py "<PDF_PATH>"

# 再執行偵測
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
record_index = 0

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
    for page_record_index, chunk in enumerate(chunks):
        records.append({
            "record_index": record_index,
            "page_index": page_idx,
            "page_number": page_idx + 1,
            "page_record_index": page_record_index,
            "loc": {"item_index": current_item_index, "chunk_index": chunk_counts[ck]},
            "chapter": current_chapter,
            "text": chunk,
            "source": "pdf_text_layer"
        })
        chunk_counts[ck] += 1
        record_index += 1

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
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py check-command -- \
  python3 ~/.codex/skills/pdf-ocr/extract_pages.py \
  "<PDF_PATH>" \
  "<PNG_DIR>" \
  --dpi <DPI>

python3 ~/.codex/skills/pdf-ocr/extract_pages.py \
  "<PDF_PATH>" \
  "<PNG_DIR>" \
  --dpi <DPI>
```

- `PNG_DIR` = 若進度中已有 `png_dir` 且目錄存在則沿用，否則用 `/tmp/pdf-ocr-<uuid4前8碼>/`
- 腳本輸出最後三行包含 `TOTAL:<n>`、`EXTRACTED:<n>`、`OUT_DIR:<path>`

渲染完成後立刻寫 manifest：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py write-manifest \
  "<PDF_PATH>" \
  "<OUTPUT_PATH>" \
  --png-dir "<PNG_DIR>" \
  --total-pages <TOTAL_PAGES>
```

> extract_pages.py 只負責 PDF→PNG 渲染（PyMuPDF 純圖像轉換），**不做任何 OCR**。

非 `--resume` 時初始化進度結構（見 §進度結構）。`--resume` 則讀取既有 `PROGRESS_FILE`。進度檔不可保存整本 records，全文只寫入 `OUTPUT_PATH`。

---

## Step 2：決定本次 batch 範圍

```
已完成 = PROGRESS.completed_pages（整數 list，0-based）
待處理 = sorted([i for i in range(TOTAL_PAGES) if i not in 已完成])
本次 batch = 待處理[:BATCH_SIZE]  # 預設 8，長時間任務不可超過 10
```

若 `本次 batch` 為空 → 跳到 Step 5

告知使用者：「本次處理第 {min+1}–{max+1} 頁（共 {len(batch)} 頁）...」

若使用 subagent/fresh worker 模式，主 agent 在此步建立 worker 任務包，且 `fork_context:false`。主 agent 不處理頁面文字，也不等待 worker 貼全文，只接收短狀態回報。

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

SKIP 頁也必須 append 一筆 record：

```json
{
  "record_index": 12,
  "page_index": 5,
  "page_number": 6,
  "page_record_index": 0,
  "loc": {"item_index": 2, "chunk_index": 0},
  "chapter": "第二章：...",
  "text": "",
  "source": "llm_vision",
  "page_sha256": "<sha256>",
  "skip": true,
  "skip_reason": "封面/空白/純裝飾"
}
```

### 3c. 切 chunks

≤ 500 字，優先在段落 `\n` 或句號 `。` 邊界切割。

### 3d. 建立 records

```json
{
  "record_index": <global_record_index>,
  "page_index": <page_idx>,
  "page_number": <page_idx + 1>,
  "page_record_index": <page_record_index>,
  "loc": {"item_index": <current_item_index>, "chunk_index": <ci>},
  "chapter": "<current_chapter>",
  "text": "<chunk_text>",
  "source": "llm_vision",
  "page_sha256": "<manifest.pages[page_idx].page_sha256>"
}
```

**`source` 必填且必為 `"llm_vision"`**（§C 規則）。
**頁序欄位必填且必須符合 §C2**。每 append 一筆前，先讀 `OUTPUT_PATH` 最後一筆確認下一個 `record_index`。

### 3e. 更新進度並立即存檔

每頁完成後先 append records 到 `OUTPUT_PATH`，再更新 `PROGRESS`：
- `completed_pages` 加入 `page_idx`
- `current_chapter`、`current_item_index`、`item_chunk_counts` 更新
- `last_record_index` 更新為最後一筆 `record_index`
- `page_hashes[page_idx]` 記錄本頁 `page_sha256`

立即寫入 `PROGRESS_FILE`（每頁完成都要存）。

禁止把本頁全文貼回主對話；需要人工抽檢時，只讀 `OUTPUT_PATH` 的前 3 筆、末 3 筆或指定頁附近 records。

---

## Step 4：回報進度

每完成 5 頁，輸出一行：
```
[15/30] 頁 16-20：已完成，累計 87 records，目前章節：第三章：命宮
```

---

## Step 5：寫出 JSONL + 健全性檢查

LLM Vision 路徑採 append-only，Step 5 不再從 `PROGRESS.records` 重寫整本 JSONL。只做全檔驗證：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py validate-jsonl "<OUTPUT_PATH>"
```

若本批完成後尚未全書完成，也必須立即跑一次；這是 batch 邊界的硬性品質門檻。

驗證失敗即 §C 違規，整批作廢重做。

### 5b. 輸出健全性快檢

先執行 guard 驗證：

```bash
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py validate-jsonl "<OUTPUT_PATH>"
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py finalize-audit "<OUTPUT_PATH>" --total-pages <TOTAL_PAGES>
```

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
📄 累計 {total_records} records，JSONL 已 append：{OUTPUT_PATH}
📌 剩餘 {remaining} 頁，請執行：
   /pdf-ocr {PDF_PATH} --resume
```

**若全部完成（§G 清理）**：

```bash
# 1. 刪 PROGRESS_FILE
rm -f "{OUTPUT_PATH}.ocr-progress.json"

# 2. 若保留 PDF，才可刪 PNG 暫存；若要刪 PDF，先 archive PNG
python3 ~/.codex/skills/pdf-ocr/pdf_ocr_guard.py deletion-gate \
  "{OUTPUT_PATH}" \
  --total-pages <TOTAL_PAGES> \
  --keep-pages

# 3. 依 deletion-gate 結果刪 PDF 或清理暫存
```

回報：
```
🎉 完成！
📄 共 {total_pages} 頁，{total_records} records
🔧 路徑：{TYPE：text 直取 / 掃描型 LLM Vision}
💾 輸出：{OUTPUT_PATH}
✅ 來源驗證：所有 record source ∈ {llm_vision, pdf_text_layer}
✅ 順序驗證：record_index/page_index/page_record_index 全部連續
✅ 覆蓋驗證：每一頁都有 record 或 skip record
✅ 證據驗證：manifest/final-report/deletion-report hash chain 已保存
🧹 暫存/證據保存策略已完成
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
  "last_record_index": 11,
  "page_hashes": {"0": "<sha256>", "1": "<sha256>"},
  "records_written": 12,
  "last_completed_at": "2026-05-07T12:00:00+08:00",
  "last_worker": "worker-id-or-session-note"
}
```

文字直取路徑一次跑完，不需要 progress 檔。LLM Vision 路徑的 progress 檔只保存狀態；若發現 progress 內保存完整 records，必須先遷移為 append-only 再繼續。

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
