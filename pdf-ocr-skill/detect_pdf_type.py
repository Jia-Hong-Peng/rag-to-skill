#!/usr/bin/env python3
"""detect_pdf_type.py: 判斷 PDF 是文字型還是掃描型（避免浪費 LLM Vision token）

用法：
  python3 detect_pdf_type.py <pdf_path>

輸出（stdout，可被 shell 解析）：
  TOTAL_PAGES:<n>
  SAMPLED_PAGES:<n>
  TOTAL_TEXT_CHARS:<n>
  AVG_CHARS_PER_PAGE:<float>
  TYPE:<text|scanned|borderline>
  RECOMMENDATION:<use_text_extraction|use_llm_vision_ocr|inspect_manually>

判斷邏輯：
  - 抽樣前 10 頁與末 10 頁（去重），讀取每頁文字層字元數
  - 平均 ≥ 50 字元/頁 → text（走文字直取）
  - 平均 < 30 字元/頁 → scanned（走 LLM Vision）
  - 30 ≤ 平均 < 50 → borderline（建議目視抽樣 1-2 頁再決定）
"""

import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("缺少 PyMuPDF，請執行：pip install pymupdf")


def main():
    if len(sys.argv) != 2:
        sys.exit("用法：detect_pdf_type.py <pdf_path>")

    pdf = Path(sys.argv[1])
    if not pdf.exists():
        sys.exit(f"找不到 PDF：{pdf}")

    doc = fitz.open(str(pdf))
    total_pages = len(doc)

    # 抽樣：前 10 頁 + 末 10 頁，去重
    head = list(range(min(10, total_pages)))
    tail = list(range(max(0, total_pages - 10), total_pages))
    indices = sorted(set(head + tail))

    total_chars = 0
    for i in indices:
        text = doc[i].get_text().strip()
        total_chars += len(text)

    sampled = len(indices)
    avg = total_chars / sampled if sampled else 0.0
    doc.close()

    if avg >= 50:
        ptype = "text"
        rec = "use_text_extraction"
    elif avg < 30:
        ptype = "scanned"
        rec = "use_llm_vision_ocr"
    else:
        ptype = "borderline"
        rec = "inspect_manually"

    print(f"TOTAL_PAGES:{total_pages}")
    print(f"SAMPLED_PAGES:{sampled}")
    print(f"TOTAL_TEXT_CHARS:{total_chars}")
    print(f"AVG_CHARS_PER_PAGE:{avg:.1f}")
    print(f"TYPE:{ptype}")
    print(f"RECOMMENDATION:{rec}")


if __name__ == "__main__":
    main()
