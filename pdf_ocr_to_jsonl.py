#!/usr/bin/env python3
"""
pdf_ocr_to_jsonl.py：掃描版 PDF（無文字層）→ JSONL（LLM Vision）

本腳本**不使用任何 Anthropic API 或 SDK**，僅負責：
1. 將 PDF 轉為 PNG 圖片（extract_pages.py）
2. 輸出「可直接貼入 Claude Code session」的 LLM Vision 指令批次
3. 最後將收集到的純文字 OCR結果 寫入 JSONL

做法：讓 Claude Code 用 Read tool 逐頁讀取 PNG，模型以視覺能力自動辨識文字，
消耗當前 session token，無需任何 API key。

用法：
  python3 pdf_ocr_to_jsonl.py <input.pdf> [--batch N] [--dpi N]
  # → 輸出引導指令，貼入 Claude Code session 執行

  python3 pdf_ocr_to_jsonl.py <input.pdf> --collect
  # → 在已執行過 OCR 的 session 中，收集結果並寫入 JSONL

依賴：pip install pymupdf（只用於 PDF → PNG，無 SDK 依賴）
"""

import sys, json, re, argparse, time
from pathlib import Path

try:
    import fitz
except ImportError:
    sys.exit("缺少 PyMuPDF，請執行：pip install pymupdf")


# ── 常數 ──────────────────────────────────────────────────────────────────────

DEFAULT_DPI   = 120
DEFAULT_BATCH = 30
DEFAULT_CHUNK = 500


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def is_blank(page) -> bool:
    """快速偵測空白頁（全白）"""
    pix = page.get_pixmap(matrix=fitz.Matrix(0.12, 0.12))
    s = pix.samples
    if not s:
        return True
    return sum(b > 245 for b in s) / len(s) > 0.97


def render_page(doc, page_idx: int, dpi: int, png_dir: Path) -> Path:
    """將指定頁面渲染為 PNG，返回 PNG 檔案路徑"""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    path = png_dir / f"page_{page_idx:04d}.png"
    if not path.exists():
        pix = doc[page_idx].get_pixmap(matrix=mat)
        pix.save(str(path))
    return path


def generate_vision_prompt(pages: list[int], png_dir: Path, output_path: Path,
                            progress_file: Path, batch_start: int) -> str:
    """產生 LLM Vision OCR 的 Claude Code 批次指令（prompt script）"""

    page_list = ", ".join(str(p + 1) for p in pages)

    prompt = f'''\
# LLM Vision OCR 任務：處理第 {page_list} 頁

## 設定
PNG 目錄：{png_dir}
JSONL 輸出：{output_path}
進度檔：{progress_file}
批次起始：{batch_start}

---

## 任務說明

請使用 Read tool 依序讀取以下 PNG 圖片：
{"".join(f'Read("{png_dir}/page_{p:04d}.png")\n')}

每次 Read tool 會自動以視覺能力辨識圖片中的文字。請完成所有頁面的讀取。

---

## 轉錄格式

讀取每頁後，請以 JSON 行輸出轉錄結果（直接 AppendWrite，不可只放在thinking裡）：

格式：
```json
{{"page": <page_number>, "chapter": "<章節名（若有）>", "text": "<轉錄文字，500字以內>"}}
```

## 判斷規則

1. **章節標題頁**：輸出 `{"page": N, "chapter": "第X章：章節名", "text": ""}`
2. **空白頁/封面/版權頁**：跳過，不輸出
3. **一般文字頁**：完整轉錄
4. **圖示頁（有知識性內容）**：用文字描述圖片內容

---

完成所有頁面後，AppendWrite 寫入：
```
[OCR批次完成] 第 {pages[0]+1}–{pages[-1]+1} 頁已完成，請執行下一批次或合併 JSONL。
```

---

## 合併 JSONL（全部完成後執行）

所有批次完成後，請將所有 JSON 行合併為標準 JSONL 格式寫入 output_path。
確認所有 record 都有 "loc" 結構：
```json
{{"loc": {{"item_index": <n>, "chunk_index": <n>}}, "chapter": "...", "text": "..."}}
```
'''
    return prompt


def estimate_pages(pdf_path: Path, dpi: int):
    """快速估算總頁數和有效頁"""
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    valid = [i for i in range(total) if not is_blank(doc[i])]
    blank = total - len(valid)
    doc.close()
    return total, valid, blank


# ── 主流程 ────────────────────────────────────────────────────────────────────

def prepare_ocr(pdf_path: Path, output_path: Path,
                dpi: int = DEFAULT_DPI, batch_size: int = DEFAULT_BATCH):
    """
    Phase 1：準備工作
    - 渲染所有頁面為 PNG
    - 掃描有效頁面
    - 輸出 LLM Vision 批次指令
    """
    pdf_path = Path(pdf_path).expanduser()
    output_path = Path(output_path).expanduser().with_suffix(".jsonl")
    png_dir = Path(f"/tmp/pdf-ocr-{pdf_path.stem}")

    print(f"\n{'='*60}")
    print(f"  PDF → LLM Vision OCR（不使用任何 API）")
    print(f"{'='*60}\n")
    print(f"PDF：{pdf_path}")
    print(f"輸出：{output_path}")
    print(f"PNG：{png_dir}")
    print(f"DPI：{dpi}，批次大小：{batch_size} 頁/批")
    print()

    # 渲染所有頁面
    print(f"正在渲染 PNG 圖片...", flush=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    total = len(doc)

    rendered = []
    for i in range(total):
        p = render_page(doc, i, dpi, png_dir)
        rendered.append(i)
        if (i + 1) % 50 == 0:
            print(f"  已渲染 {i+1}/{total} 頁", flush=True)
    doc.close()
    print(f"✓ 完成，共 {total} 頁\n")

    # 掃描有效頁
    print(f"正在掃描有效頁面...", flush=True)
    doc2 = fitz.open(str(pdf_path))
    valid_pages = [i for i in range(total) if not is_blank(doc2[i])]
    doc2.close()
    blank_count = total - len(valid_pages)
    print(f"有效頁面：{len(valid_pages)}（空白 {blank_count} 頁跳過）\n")

    # 寫入進度檔（初期狀態）
    progress_file = output_path.with_suffix(".ocr-progress.json")
    progress = {
        "pdf_path": str(pdf_path),
        "output_path": str(output_path),
        "png_dir": str(png_dir),
        "total_pages": total,
        "valid_pages": valid_pages,
        "completed_pages": [],
        "batch_count": 0,
        "batches": [],
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    # 切割批次
    batches = []
    for i in range(0, len(valid_pages), batch_size):
        batch = valid_pages[i:i+batch_size]
        batches.append((i // batch_size, batch))

    print(f"共 {len(batches)} 個批次：\n")
    for idx, (_, batch) in enumerate(batches):
        pages_str = f"第 {batch[0]+1}–{batch[-1]+1} 頁（共 {len(batch)} 頁）"
        print(f"  批次 {idx+1}/{len(batches)}：{pages_str}")

    print()
    print(f"{'='*60}")
    print(f"  下一步：複製以下指令到 Claude Code session 執行")
    print(f"{'='*60}\n")

    for idx, (_, batch) in enumerate(batches):
        print(f"\n{'─'*60}")
        print(f"# 批次 {idx+1}/{len(batches)}：{'，'.join(f'頁{p+1}' for p in batch[:3])}{'...' if len(batch) > 3 else ''}")
        print(f"{'─'*60}\n")
        prompt = generate_vision_prompt(batch, png_dir, output_path, progress_file, idx)
        print(prompt)

    print(f"\n{'='*60}")
    print(f"  說明")
    print(f"{'='*60}")
    print(f"""
完成所有批次後，在 Claude Code session 中執行：
→ 合併所有 JSON 行為最終 JSONL（格式見上方）

或者直接貼入 Claude Code，逐步執行每個批次的視覺辨識指令。
每次完成一個批次，Claude Code 會自動記錄進度。
""")

    return png_dir, batches, progress_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="掃描版 PDF → PNG + LLM Vision 批次 OCR 指令（無 API 依賴）"
    )
    parser.add_argument("pdf",   type=Path, help="輸入 PDF 路徑")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="輸出 JSONL 路徑（預設：同目錄，副檔名改 .jsonl）")
    parser.add_argument("--dpi",    type=int, default=DEFAULT_DPI,
                        help=f"DPI（預設 {DEFAULT_DPI}）")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                        help=f"每批次頁數（預設 {DEFAULT_BATCH}）")
    parser.add_argument("--collect", action="store_true",
                        help="收集已 OCR 的結果並寫入 JSONL")
    parser.add_argument("--merge",  type=Path, nargs="?", default=None,
                        help="手動指定進度檔並合併為 JSONL")

    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"找不到檔案：{args.pdf}")

    output_path = args.output or args.pdf.with_suffix(".jsonl")

    if args.merge:
        # 手動合併模式（需要手動粘貼 OCR 結果）
        prog = Path(args.merge)
        if prog.exists():
            with open(prog, encoding='utf-8') as f:
                p = json.load(f)
            print(f"進度檔：{prog}")
            print(f"已完成頁面：{len(p.get('completed_pages', []))}")
            print(f"批次數：{p.get('batch_count', 0)}")
        return

    if args.collect:
        print("收集模式需要在 Claude Code session 中手動粘貼結果...")
        return

    # 預設：準備工作 + 輸出批次指令
    prepare_ocr(args.pdf, output_path, dpi=args.dpi, batch_size=args.batch)


if __name__ == "__main__":
    main()