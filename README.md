# rag-to-skill

> 把任何 JSONL 格式的 RAG 源文件，轉換為結構完整、可驗證、可分享的 [Claude Code Skill](https://docs.anthropic.com/en/docs/claude-code/skills)。

## 核心承諾

**100% 完成度** — 每個 JSONL entity 都有對應 reference，每個斷言都有 RAG 錨點，無 AI 編造。

## 安裝

```bash
cp -r rag-to-skill ~/.claude/skills/rag-to-skill
```

安裝後，在 Claude Code 中輸入 `/rag-to-skill` 即可觸發。

## 七階段工作流程

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

| 階段 | 產出 | 完成條件 |
|---|---|---|
| 1. SCAN | schema 報告、entity 清單 | 知道有幾個 item、主要章節 |
| 2. PLAN | ref 檔分組方案 | 每個 entity 都有歸屬 ref 檔 |
| 3. SCAFFOLD | 空的 skill.md + references/ | 目錄結構存在，frontmatter 完整 |
| 4. BUILD | 每個 ref 檔填滿 RAG 原文 | 所有 entity 都有完整內容 |
| 5. ANCHOR | 每個斷言都有 `> 來源:...` 標注 | 錨點密度 ≥ 15/1000 行 |
| 6. AUDIT | validate.py 全綠 | 無 SEVERE 問題 |
| 7. SHIP | 安裝到 `~/.claude/skills/` | skill 可被觸發使用 |

## 典型觸發語句

- 「把 paidRag/大耕老師/紫微攻略.jsonl 做成 skill」
- 「幫我從這本書建立一個 skill，100% 完成度」
- 「這個 skill 還沒完整，幫我對齊 RAG」
- 「這個 skill 要給同事用，準備提交」

## Reference 目錄

| 檔案 | 何時讀 |
|---|---|
| `references/01-quickstart.md` | 已熟悉流程，想快速開始 |
| `references/02-jsonl-scan.md` | 拿到 JSONL 的第一件事 |
| `references/03-skill-scaffold.md` | 建立 skill 目錄結構時 |
| `references/04-content-build.md` | 填充 reference 內容時 |
| `references/05-anchor-protocol.md` | 在 ref 檔中加入來源標注時 |
| `references/06-validation-scripts.md` | 執行品質驗證前（含 validate.py v1.2） |
| `references/07-completion-checklist.md` | 準備提交或分享 skill 前 |
| `references/08-case-example.md` | 第一次用本 skill、想看完整案例 |

## 內建驗證工具（validate.py v1.2）

從 `references/06-validation-scripts.md` 一鍵提取：

```bash
python3 << 'EXTRACT'
import re, os, sys
path = os.path.expanduser('~/.claude/skills/rag-to-skill/references/06-validation-scripts.md')
content = open(path, encoding='utf-8').read()
m = re.search(r'```python\n(.*?)```', content, re.DOTALL)
if not m:
    sys.exit('ERROR: 無法找到 python code block')
open('validate.py', 'w', encoding='utf-8').write(m.group(1))
print(f'validate.py 已提取（{len(m.group(1).splitlines())} 行）')
EXTRACT

python3 validate.py <skill-dir>
python3 validate.py <skill-dir> <path/to/source.jsonl>   # 含 JSONL 覆蓋率
```

validate.py 執行 5 項自動檢查：frontmatter 完整性、空骨架、內部連結、RAG 錨點密度、佔位符殘留。

## 前置需求

- `python3`（標準函式庫即可，無需額外安裝）
- JSONL 格式的 RAG 源文件（每行一個 JSON 物件）
- Claude Code CLI

## 電子書 → JSONL 轉換工具

### PDF

```bash
pip install pymupdf

python3 pdf_to_jsonl.py 紫微攻略.pdf
python3 pdf_to_jsonl.py 紫微攻略.pdf ~/paidRag/紫微攻略.jsonl
python3 pdf_to_jsonl.py 紫微攻略.pdf --chunk-size 800
python3 pdf_to_jsonl.py 紫微攻略.pdf --no-toc   # TOC 不完整時，每頁一個 item
```

| 功能 | 說明 |
|---|---|
| TOC 書籤 | 自動用 PDF 書籤分章，書籤第一章前的頁面歸入「前言 / 版權頁」 |
| Header/Footer 偵測 | 分析前 30 頁，過濾重複出現的頁首/頁尾（書名行、頁碼）|
| 無書籤 fallback | 自動改為每頁一個 item |

### EPUB

EPUB 電子書可用 `epub_to_jsonl.py` 直接轉成相容格式：

```bash
# 安裝依賴
pip install ebooklib beautifulsoup4

# 轉換
python3 epub_to_jsonl.py 紫微攻略.epub
# → 輸出 紫微攻略.jsonl

# 指定輸出路徑
python3 epub_to_jsonl.py 紫微攻略.epub ~/paidRag/紫微攻略.jsonl

# 調整 chunk 大小（預設 500 字元）
python3 epub_to_jsonl.py 紫微攻略.epub --chunk-size 800
```

輸出格式：

```json
{"loc": {"item_index": 0, "chunk_index": 0}, "chapter": "第一章：命宮", "text": "..."}
{"loc": {"item_index": 0, "chunk_index": 1}, "chapter": "第一章：命宮", "text": "..."}
{"loc": {"item_index": 1, "chunk_index": 0}, "chapter": "第二章：財帛宮", "text": "..."}
```

轉換完成後，直接用 `rag-to-skill` 建立 skill：在 Claude Code 輸入「把 xxx.jsonl 做成 skill」即可。

## 授權

MIT
