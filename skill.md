---
name: rag-to-skill
description: |
  從 JSONL（RAG 源文件）建立 Claude Code Skill 的完整工作流程工具。
  涵蓋「從零到 100% 完成度」的七個階段：JSONL 掃描 → 架構規劃 → 骨架生成
  → 內容填充 → RAG 錨點標注 → 驗證稽核 → 最終交付。
  本 skill 完全獨立，不依賴任何特定 repo 的 CI 腳本，可帶到任意專案使用。
  何時觸發：
  - 使用者說「把這個 JSONL/RAG 轉成 skill」
  - 使用者說「幫我從這本書建立一個 skill」
  - 使用者有 .jsonl 檔案並想建立可分享的 Claude Code skill
  - 使用者說「建立新的 skill，來源是 XXX」
  - 使用者說「我要把 paidRag 裡面的書做成技能」
  適用對象：skill 開發者、知識庫維護者、想把書/文件轉成可執行 AI 知識的任何人。
  典型提問：
  - 「把 紫微攻略.jsonl 做成 skill」
  - 「幫我從 RAG 建一個 skill，100% 完成度」
  - 「這本書的 JSONL 要怎麼變成可以給別人用的技能？」
---

# RAG → Skill 建構工具

> 本 skill 是**流程工具**，不是領域知識。
> 目標：把任何 JSONL 格式的 RAG 源文件，轉換為結構完整、可驗證、可分享的 Claude Code Skill。
> 核心承諾：**100% 完成度** — 每個 JSONL entity 都有對應 reference，每個斷言都有 RAG 錨點，無 AI 編造。

---

## 一、適用前提

本 skill 需要：

1. **JSONL 來源文件**：任何 JSON Lines 格式，每行一個 JSON 物件
2. **目標輸出目錄**：放置新 skill 的路徑（可是任意 repo 或 `~/.claude/skills/`）
3. **基本 CLI 環境**：`python3`、`bash`、`git`（可選）

若無現成 JSONL，先用以下工具轉換，再回來：

**重要：若來源檔同目錄已存在同名 `.jsonl`，預設必須直接使用既有 JSONL，不可另轉一份到 `ocr-jsonl/` 或其他集中目錄。** 只有使用者明確要求重做或覆蓋時，才可重新轉換；重新轉換前必須先說明會耗費 token/時間，並保留舊檔。

| 工具 | 支援格式 | 指令 |
|---|---|---|
| `any_to_jsonl.py` | **萬用**（PDF / EPUB / DOCX / PPTX / HTML...） | `python3 any_to_jsonl.py <file>` |
| `pdf_to_jsonl.py` | PDF（有文字層，TOC 分章） | `python3 pdf_to_jsonl.py <file.pdf>` |
| `epub_to_jsonl.py` | EPUB（spine 順序 + 標題萃取） | `python3 epub_to_jsonl.py <file.epub>` |
| `pdf_ocr_to_jsonl.py` | **掃描版 PDF**（無文字層，LLM Vision OCR） | `python3 pdf_ocr_to_jsonl.py <file.pdf>` |

腳本位置：`~/.claude/skills/rag-to-skill/`（或同 repo 根目錄）。
依賴：`pip install markitdown`（any）、`pip install pymupdf`（pdf/ocr）、`pip install ebooklib beautifulsoup4`（epub）。
OCR：使用 Claude Code Read tool 視覺能力，**不需要** Anthropic API key，不使用任何 MCP。

---

## 二、七階段工作流程

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 1.SCAN   │ → │ 2.PLAN   │ → │ 3.SCAFFOLD│ → │ 4.BUILD  │
│ JSONL分析 │   │ 架構規劃  │   │ 骨架生成  │   │ 內容填充  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                    │
┌──────────┐   ┌──────────┐   ┌──────────┐         │
│ 7.SHIP   │ ← │ 6.AUDIT  │ ← │ 5.ANCHOR │ ←───────┘
│ 交付/提交 │   │ 驗證稽核  │   │ 錨點標注  │
└──────────┘   └──────────┘   └──────────┘
```

| 階段 | 輸入 | 產出 | 完成條件 |
|---|---|---|---|
| 1. SCAN | JSONL 路徑 | schema 報告、entity 清單、章節樹 | 知道有幾個 item、schema 欄位、主要章節 |
| 2. PLAN | entity 清單 | reference 檔分組方案、skill.md 大綱 | 每個 entity 都有歸屬 ref 檔 |
| 3. SCAFFOLD | 分組方案 | 空的 skill.md + references/ 目錄 + 空 ref 檔 | 目錄結構存在，frontmatter 完整 |
| 4. BUILD | JSONL + ref 空檔 | 每個 ref 檔填滿 RAG 原文（忠實，不縮寫） | 所有 entity 都有完整內容 |
| 5. ANCHOR | 填完的 ref 檔 | 每個斷言都有 `> 來源:...` 標注 | 錨點密度 ≥15/1000 行 |
| 6. AUDIT | 完整 skill | 驗證報告、無 AI 編造確認 | validate.py 全綠 |
| 7. SHIP | 驗證通過的 skill | 安裝到 `~/.claude/skills/` 或提交 PR | skill 可被觸發使用 |

---

## 三、Reference 目錄

| 檔案 | 內容 | 何時讀 |
|---|---|---|
| `references/01-quickstart.md` | 5 分鐘快速啟動：完整指令序列，最簡情境 | 已熟悉流程，想快速開始 |
| `references/02-jsonl-scan.md` | 階段 1：JSONL 結構探索、schema 發現、entity 清單生成 | 拿到 JSONL 的第一件事 |
| `references/03-skill-scaffold.md` | 階段 2-3：架構規劃 + skill 目錄骨架生成（含 skill.md 模板） | 開始建立 skill 目錄結構時 |
| `references/04-content-build.md` | 階段 4：從 JSONL 提取原文 → 寫入 ref 檔（含忠實原則） | 填充 reference 內容時 |
| `references/05-anchor-protocol.md` | 階段 5：RAG 錨點格式規範、寫法範例、常見錯誤 | 在 ref 檔中加入來源標注時 |
| `references/06-validation-scripts.md` | 階段 6：validate.py 獨立腳本（可複製到任何 repo），驗證邏輯 | 執行品質驗證前 |
| `references/07-completion-checklist.md` | 100% 完成度判準（Severe/Medium/Light）+ 交付前 checklist | 準備提交或分享 skill 前 |
| `references/08-case-example.md` | 完整案例：ziwei 專案 紫微攻略.jsonl → skill 實戰走查 | 第一次用本 skill、想看完整案例 |

---

## 四、典型呼叫流程

### 情境 A：從零開始建新 skill

```
使用者：「把 paidRag/大耕老師/紫微攻略.jsonl 做成 skill」

執行順序：
1. 讀 references/01-quickstart.md → 確認 5 步指令
2. 階段 1：按 references/02-jsonl-scan.md 掃描 JSONL
3. 階段 2-3：按 references/03-skill-scaffold.md 規劃 + 建目錄
4. 階段 4：按 references/04-content-build.md 逐 entity 填充
5. 階段 5：按 references/05-anchor-protocol.md 加錨點
6. 階段 6：按 references/06-validation-scripts.md 執行 validate.py
7. 階段 7：按 references/07-completion-checklist.md 確認後安裝
```

### 情境 B：已有 skill，要提升到 100% 完成度

```
使用者：「這個 skill 還沒完整，幫我對齊 RAG」

→ 跳過階段 1-3
→ 先執行 references/04-content-build.md §4.6 孤兒檢查，確認哪些 item 未被覆蓋
→ 從階段 4 補缺漏 entity 開始（references/04-content-build.md）
→ 重點讀 references/07-completion-checklist.md 找缺口
→ 執行 validate.py 找具體問題
```

### 情境 C：準備提交到 skill hub

```
使用者：「這個 skill 要給同事用，準備提交」

→ 直接讀 references/07-completion-checklist.md
→ 確認 skill 獨立性（不依賴外部 repo 路徑）
→ 執行完整 validate.py
→ 按 references/07-completion-checklist.md §六 獨立性 checklist
```

---

## 五、快速判斷：我現在在哪個階段？

```
有 JSONL，但還沒有 skill 目錄？
  → 從階段 1 開始（references/02-jsonl-scan.md 掃描 JSONL）
  → 熟練後可直接看 references/01-quickstart.md（壓縮指令版）

有 skill 目錄，但 references/ 是空的或不完整？
  → 從階段 4 開始（references/04-content-build.md）

skill 已填充，但缺來源標注？
  → 階段 5（references/05-anchor-protocol.md）

skill 看起來完整，要驗證？
  → 直接執行 references/06-validation-scripts.md 的 validate.py

要提交/分享？
  → references/07-completion-checklist.md §六 獨立性 checklist
```
