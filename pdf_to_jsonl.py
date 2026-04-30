#!/usr/bin/env python3
"""
pdf_to_jsonl：PDF 電子書轉 JSONL

用法：
  python3 pdf_to_jsonl.py <input.pdf>
  python3 pdf_to_jsonl.py <input.pdf> <output.jsonl>
  python3 pdf_to_jsonl.py <input.pdf> --chunk-size 800
  python3 pdf_to_jsonl.py <input.pdf> --no-toc      # 忽略書籤，每頁一個 item

輸出 schema 相容 rag-to-skill（item_index / chunk_index / chapter / text）。

依賴：
  pip install pymupdf
"""

import sys, json, re, argparse
from pathlib import Path
from collections import Counter

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("缺少 PyMuPDF，請執行：pip install pymupdf")


# ── 常數 ──────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 500      # 每個 chunk 最大字元數
NOISE_SAMPLE_PAGES = 30       # 偵測 header/footer 的取樣頁數
NOISE_APPEAR_RATE  = 0.3      # 出現率 > 30% → 視為雜訊
NOISE_MAX_LEN      = 60       # 超過此長度不視為 header/footer


# ── Header / Footer 偵測 ─────────────────────────────────────────────────────

def detect_noise(doc):
    """
    分析前 N 頁的文字區塊位置，找出重複出現在頁面頂部或底部的短文字
    （書名行、頁碼等），返回需要過濾的字串 set。
    """
    top_counter = Counter()
    bot_counter = Counter()
    sample = min(NOISE_SAMPLE_PAGES, len(doc))

    for i in range(sample):
        page = doc[i]
        h = page.rect.height
        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            text = block[4].strip()
            if not text or len(text) > NOISE_MAX_LEN:
                continue
            y_mid = (block[1] + block[3]) / 2
            if y_mid < h * 0.10:
                top_counter[text] += 1
            elif y_mid > h * 0.90:
                bot_counter[text] += 1

    threshold = sample * NOISE_APPEAR_RATE
    noise = set()
    for text, cnt in {**top_counter, **bot_counter}.items():
        if cnt >= threshold:
            noise.add(text)
    return noise


# ── 頁面文字萃取 ──────────────────────────────────────────────────────────────

def page_to_paragraphs(page, noise):
    """
    提取單頁文字，過濾 header/footer 與純頁碼行，
    返回段落字串列表。
    """
    raw = page.get_text("text")

    # 逐行過濾雜訊
    clean = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            clean.append("")          # 保留空行作為段落分隔
            continue
        if s in noise:
            continue
        if re.fullmatch(r"\d{1,4}", s):   # 純頁碼
            continue
        clean.append(line)

    text = "\n".join(clean)

    # 段落切分：以連續空行為邊界
    paragraphs = []
    for para in re.split(r"\n{2,}", text):
        # 段落內換行合併為空格（處理 PDF 的硬換行）
        para = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", para)
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)

    return paragraphs


# ── 切塊 ──────────────────────────────────────────────────────────────────────

def chunk_paragraphs(paragraphs, max_chars):
    """
    合併段落成 chunks，每個 chunk ≤ max_chars。
    超長單段落按句子邊界切割（支援中文句號、省略號）。
    """
    chunks = []
    buf, buf_len = [], 0

    def flush():
        nonlocal buf, buf_len
        t = "\n".join(buf).strip()
        if t:
            chunks.append(t)
        buf.clear()
        buf_len = 0

    for block in paragraphs:
        if len(block) > max_chars:
            if buf:
                flush()
            sentences = re.split(r"(?<=[。！？…\.!?])\s*", block)
            s_buf, s_len = [], 0
            for sent in sentences:
                if s_len + len(sent) > max_chars and s_buf:
                    chunks.append("".join(s_buf).strip())
                    s_buf, s_len = [], 0
                s_buf.append(sent)
                s_len += len(sent)
            if s_buf:
                chunks.append("".join(s_buf).strip())
        else:
            if buf_len + len(block) + 1 > max_chars and buf:
                flush()
            buf.append(block)
            buf_len += len(block) + 1

    if buf:
        flush()

    return [c for c in chunks if c]


# ── TOC → 章節分組 ────────────────────────────────────────────────────────────

def build_chapters(doc, use_toc):
    """
    依 TOC 書籤把頁面索引分組：[(chapter_title, [page_indices])]

    - 有書籤時：取最淺 level 為主章節，每章包含其頁面範圍。
      前置頁面（TOC 第一章之前）統一歸入「前言」。
    - 無書籤或 use_toc=False：每頁一個 item。
    """
    total = len(doc)

    if not use_toc:
        return [(f"第 {i + 1} 頁", [i]) for i in range(total)]

    toc = doc.get_toc(simple=True)   # [(level, title, page_1based), ...]

    if not toc:
        return [(f"第 {i + 1} 頁", [i]) for i in range(total)]

    # 取最淺 level
    min_level = min(t[0] for t in toc)
    main = [
        (title.strip() or "（無標題）", max(0, page - 1))
        for level, title, page in toc
        if level == min_level and page > 0
    ]

    if not main:
        return [(f"第 {i + 1} 頁", [i]) for i in range(total)]

    chapters = []

    # 前置頁面（第一章書籤之前）
    if main[0][1] > 0:
        chapters.append(("前言 / 版權頁", list(range(0, main[0][1]))))

    for i, (title, start) in enumerate(main):
        end = main[i + 1][1] - 1 if i + 1 < len(main) else total - 1
        pages = list(range(start, end + 1))
        if pages:
            chapters.append((title, pages))

    return chapters


# ── 主流程 ────────────────────────────────────────────────────────────────────

def pdf_to_jsonl(pdf_path, output_path, chunk_size, use_toc=True, verbose=True):
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # 書名
    meta = doc.metadata or {}
    book_title = (meta.get("title") or "").strip() or pdf_path.stem

    # 偵測雜訊
    noise = detect_noise(doc)
    if verbose and noise:
        print(f"過濾 header/footer：{noise}")

    # 章節分組
    toc_count = len(doc.get_toc(simple=True))
    chapters = build_chapters(doc, use_toc)

    total_items  = 0
    total_chunks = 0
    skipped      = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for item_index, (chapter_title, page_indices) in enumerate(chapters):

            # 合併章節內所有頁的段落
            all_paragraphs = []
            for pg_i in page_indices:
                all_paragraphs.extend(page_to_paragraphs(doc[pg_i], noise))

            if not all_paragraphs:
                skipped += 1
                continue

            chunks = chunk_paragraphs(all_paragraphs, chunk_size)
            if not chunks:
                skipped += 1
                continue

            for chunk_index, text in enumerate(chunks):
                record = {
                    "loc": {
                        "item_index": item_index,
                        "chunk_index": chunk_index,
                    },
                    "chapter": chapter_title,
                    "text": text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_items += 1

    doc.close()

    if verbose:
        print(f"書名      ：{book_title}")
        print(f"總頁數    ：{total_pages}")
        print(f"TOC 書籤  ：{toc_count} 個")
        print(f"章節分組  ：{len(chapters)} 個（跳過空白 {skipped} 個）")
        print(f"有效 items：{total_items}")
        print(f"總 chunks ：{total_chunks}")
        print(f"輸出      ：{output_path}")

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PDF 電子書 → JSONL（rag-to-skill 相容格式）"
    )
    parser.add_argument("pdf", type=Path, help="輸入 PDF 路徑")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="輸出 JSONL 路徑（預設：同目錄，副檔名改 .jsonl）"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, metavar="N",
        help=f"每個 chunk 最大字元數（預設 {DEFAULT_CHUNK_SIZE}）"
    )
    parser.add_argument(
        "--no-toc", action="store_true",
        help="忽略 PDF 書籤，每頁作為獨立 item（TOC 不完整時使用）"
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"找不到檔案：{args.pdf}")
    if args.pdf.suffix.lower() != ".pdf":
        sys.exit(f"不支援的檔案格式（需要 .pdf）：{args.pdf}")

    output = args.output or args.pdf.with_suffix(".jsonl")
    pdf_to_jsonl(args.pdf, output, args.chunk_size, use_toc=not args.no_toc)


if __name__ == "__main__":
    main()
