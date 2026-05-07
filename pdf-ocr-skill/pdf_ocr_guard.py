#!/usr/bin/env python3
"""Guard checks for the pdf-ocr skill.

This script does not perform OCR. It makes the skill's hard rules executable:
- block banned OCR commands before they are run
- reject JSONL records with illegal or missing source values
- detect unfinished pdf-ocr progress files before starting another book
- reject legacy progress files that store full OCR records
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


LEGAL_SOURCES = {"llm_vision", "pdf_text_layer"}
SCHEMA_VERSION = "pdf-ocr-audit-v1"
BANNED_PATTERNS = [
    r"\btesseract\b",
    r"\bpytesseract\b",
    r"\bpaddleocr\b",
    r"\bpaddle_ocr\b",
    r"\bpaddleocr-cli\b",
    r"\bpaddle\s+ocr\b",
    r"\beasyocr\b",
    r"\bcnocr\b",
    r"\bocrmypdf\b",
    r"\bgcloud\s+vision\b",
    r"\baws\s+textract\b",
    r"\baz\s+cognitiveservices\s+vision\b",
    r"\baliyun\s+ocr\b",
    r"\btencent\s+ocr\b",
    r"\bmcp__[^ \t\r\n]*image[^ \t\r\n]*\b",
    r"\bmcp__MiniMax__understand_image\b",
]


def fail(message: str) -> None:
    print(f"BLOCKED:{message}", file=sys.stderr)
    raise SystemExit(2)


def check_command(command: str) -> None:
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            fail(f"禁止的 OCR/圖像工具命令: {pattern}")
    print("OK:command")


def first_jsonl_record(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no} 不是合法 JSON: {exc}")
    return None


def validate_existing_jsonl(path: Path, strict_missing: bool) -> None:
    record = first_jsonl_record(path)
    if record is None:
        print("OK:jsonl-empty-or-missing")
        return
    source = record.get("source")
    if source not in LEGAL_SOURCES:
        if source is None and not strict_missing:
            fail(f"{path} 第一筆缺少 source，必須人工確認後才可沿用")
        fail(f"{path} 第一筆 source={source!r} 不合法，視為已污染，不可 --resume")
    print("OK:jsonl-source")


def validate_jsonl(path: Path) -> None:
    if not path.exists():
        fail(f"找不到 JSONL: {path}")
    manifest_path = Path(str(path) + ".manifest.json")
    manifest_hashes = {}
    if manifest_path.exists():
        manifest = load_progress(manifest_path)
        if manifest is None:
            fail(f"{manifest_path} 不是合法 JSON")
        for page in manifest.get("pages") or []:
            if isinstance(page.get("page_index"), int):
                manifest_hashes[page["page_index"]] = page.get("page_sha256")
    checked = 0
    previous_record_index = -1
    previous_page_index = -1
    page_record_counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no} 不是合法 JSON: {exc}")
            source = record.get("source")
            if source not in LEGAL_SOURCES:
                fail(f"{path}:{line_no} source={source!r} 不合法")
            text = record.get("text")
            if not isinstance(text, str):
                fail(f"{path}:{line_no} text 必須是字串")
            if not text.strip() and not record.get("skip"):
                fail(f"{path}:{line_no} text 空白時必須標記 skip:true")
            if record.get("skip") and not record.get("skip_reason"):
                fail(f"{path}:{line_no} skip record 必須包含 skip_reason")

            record_index = record.get("record_index")
            page_index = record.get("page_index")
            page_number = record.get("page_number")
            page_record_index = record.get("page_record_index")
            if not isinstance(record_index, int):
                fail(f"{path}:{line_no} 缺少整數 record_index")
            if record_index != previous_record_index + 1:
                fail(f"{path}:{line_no} record_index 不連續: {record_index} after {previous_record_index}")
            if not isinstance(page_index, int) or page_index < 0:
                fail(f"{path}:{line_no} 缺少合法 page_index")
            if page_index < previous_page_index:
                fail(f"{path}:{line_no} page_index 倒退: {page_index} after {previous_page_index}")
            if page_number != page_index + 1:
                fail(f"{path}:{line_no} page_number 必須等於 page_index + 1")
            expected_page_record_index = page_record_counts.get(page_index, 0)
            if page_record_index != expected_page_record_index:
                fail(
                    f"{path}:{line_no} page_record_index 不連續: "
                    f"{page_record_index} expected {expected_page_record_index}"
                )
            if source == "llm_vision" and not record.get("page_sha256"):
                fail(f"{path}:{line_no} llm_vision record 必須包含 page_sha256")
            if page_index in manifest_hashes and record.get("page_sha256") != manifest_hashes[page_index]:
                fail(f"{path}:{line_no} page_sha256 與 manifest page {page_index + 1} 不符")
            page_record_counts[page_index] = expected_page_record_index + 1
            previous_record_index = record_index
            previous_page_index = page_index
            checked += 1
    print(f"OK:jsonl-records:{checked}")


def load_progress(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, text: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        fail(f"拒絕覆寫既有正式 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_json(path: Path, data: object, overwrite: bool = False) -> None:
    atomic_write_text(path, canonical_json(data) + "\n", overwrite=overwrite)


def check_single_book(current_output: Path | None) -> None:
    current_progress = None
    if current_output is not None:
        current_progress = Path(str(current_output) + ".ocr-progress.json").resolve()

    progress_files = [Path(p) for p in glob.glob("**/*.ocr-progress.json", recursive=True)]
    unfinished = []
    for path in progress_files:
        resolved = path.resolve()
        if current_progress is not None and resolved == current_progress:
            continue
        data = load_progress(path)
        if not data:
            continue
        total = int(data.get("total_pages") or 0)
        completed = len(set(data.get("completed_pages") or []))
        if total and completed < total:
            unfinished.append((path, data.get("pdf_path", "未知 PDF"), completed, total))

    tmp_dirs = [Path(p) for p in glob.glob("/tmp/pdf-ocr-*") if Path(p).is_dir()]
    if unfinished:
        details = "; ".join(f"{pdf} ({done}/{total}, {path})" for path, pdf, done, total in unfinished)
        fail(f"已有未完成 OCR 任務: {details}")
    if tmp_dirs and current_progress is None:
        dirs = ", ".join(str(p) for p in tmp_dirs)
        fail(f"存在 pdf-ocr 暫存目錄，請先確認或清除: {dirs}")
    print("OK:single-book")


def validate_progress(path: Path) -> None:
    if not path.exists():
        print("OK:progress-missing")
        return
    data = load_progress(path)
    if data is None:
        fail(f"{path} 不是合法 progress JSON")
    records = data.get("records")
    if records:
        fail(f"{path} 含完整 records，請先遷移為 append-only JSONL 再 resume")
    required = {"pdf_path", "output_path", "total_pages", "png_dir", "completed_pages"}
    missing = sorted(required - set(data))
    if missing:
        fail(f"{path} 缺少必要欄位: {', '.join(missing)}")
    print("OK:progress-state-only")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(pdf_path: Path, output_path: Path, png_dir: Path | None, total_pages: int | None, overwrite: bool) -> None:
    if not pdf_path.exists():
        fail(f"找不到原始 PDF: {pdf_path}")
    manifest_path = Path(str(output_path) + ".manifest.json")
    pages = []
    if png_dir is not None and png_dir.exists():
        for page_png in sorted(png_dir.glob("page_*.png")):
            match = re.search(r"page_(\d+)\.png$", page_png.name)
            if not match:
                continue
            page_index = int(match.group(1))
            pages.append(
                {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "png_path": str(page_png),
                    "page_sha256": sha256_file(page_png),
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pdf_path": str(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "total_pages": total_pages if total_pages is not None else len(pages),
        "pages": pages,
    }
    validate_manifest_data(manifest, require_originals=True)
    atomic_write_json(manifest_path, manifest, overwrite=overwrite)
    print(f"OK:manifest:{manifest_path}")


def validate_manifest_data(data: dict, require_originals: bool, total_pages: int | None = None) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        fail(f"manifest schema_version 必須是 {SCHEMA_VERSION}")
    expected_total = total_pages if total_pages is not None else data.get("total_pages")
    if not isinstance(expected_total, int) or expected_total < 0:
        fail("manifest 缺少合法 total_pages")
    pdf_path = Path(data.get("pdf_path", ""))
    if require_originals and not pdf_path.exists():
        fail(f"manifest 指向的 PDF 不存在: {pdf_path}")
    if pdf_path.exists() and sha256_file(pdf_path) != data.get("pdf_sha256"):
        fail(f"{pdf_path} SHA256 與 manifest 不符")
    pages = data.get("pages")
    if not isinstance(pages, list):
        fail("manifest pages 必須是 list")
    indices = []
    for page in pages:
        if not isinstance(page, dict):
            fail("manifest page entry 必須是 object")
        page_index = page.get("page_index")
        page_number = page.get("page_number")
        png_path = Path(page.get("png_path", ""))
        if not isinstance(page_index, int) or page_index < 0:
            fail("manifest page_index 不合法")
        if page_number != page_index + 1:
            fail(f"manifest page {page_index} page_number 不合法")
        if require_originals and not png_path.exists():
            fail(f"manifest 指向的 PNG 不存在: {png_path}")
        if png_path.exists() and sha256_file(png_path) != page.get("page_sha256"):
            fail(f"{png_path} SHA256 與 manifest 不符")
        indices.append(page_index)
    expected = list(range(expected_total))
    if sorted(indices) != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        fail(f"manifest 頁面覆蓋不完整 missing={[i + 1 for i in missing[:20]]} extra={extra[:20]}")


def validate_manifest(output_path: Path, require_originals: bool, total_pages: int | None) -> None:
    manifest_path = Path(str(output_path) + ".manifest.json")
    if not manifest_path.exists():
        fail(f"找不到 manifest: {manifest_path}")
    data = load_progress(manifest_path)
    if data is None:
        fail(f"{manifest_path} 不是合法 JSON")
    validate_manifest_data(data, require_originals=require_originals, total_pages=total_pages)
    print("OK:manifest")


def validate_coverage(output_path: Path, total_pages: int, allow_skipped: bool) -> None:
    if not output_path.exists():
        fail(f"找不到 JSONL: {output_path}")
    seen_pages: set[int] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{output_path}:{line_no} 不是合法 JSON: {exc}")
            page_index = record.get("page_index")
            if isinstance(page_index, int):
                seen_pages.add(page_index)
    expected = set(range(total_pages))
    missing = sorted(expected - seen_pages)
    extra = sorted(seen_pages - expected)
    if extra:
        fail(f"JSONL 包含超出總頁數的 page_index: {extra[:20]}")
    if missing and not allow_skipped:
        fail(f"JSONL 缺頁: {[m + 1 for m in missing[:50]]}")
    print(f"OK:coverage:pages_with_records={len(seen_pages)} missing={len(missing)}")


def hash_jsonl_pages(output_path: Path) -> dict[int, str]:
    page_lines: dict[int, list[str]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            page_index = record["page_index"]
            page_lines.setdefault(page_index, []).append(canonical_json(record))
    return {
        page_index: hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
        for page_index, lines in page_lines.items()
    }


def finalize_audit(output_path: Path, total_pages: int, overwrite: bool) -> None:
    validate_jsonl(output_path)
    validate_coverage(output_path, total_pages=total_pages, allow_skipped=False)
    validate_manifest(output_path, require_originals=True, total_pages=total_pages)
    manifest_path = Path(str(output_path) + ".manifest.json")
    final_report = {
        "schema_version": SCHEMA_VERSION,
        "output_path": str(output_path),
        "jsonl_sha256": sha256_file(output_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "total_pages": total_pages,
        "page_record_hashes": hash_jsonl_pages(output_path),
    }
    report_path = Path(str(output_path) + ".final-report.json")
    atomic_write_json(report_path, final_report, overwrite=overwrite)
    print(f"OK:final-report:{report_path}")


def deletion_gate(output_path: Path, total_pages: int, keep_pdf: bool, keep_pages: bool, overwrite: bool) -> None:
    final_report_path = Path(str(output_path) + ".final-report.json")
    manifest_path = Path(str(output_path) + ".manifest.json")
    if not final_report_path.exists():
        fail(f"刪除前缺少 final report: {final_report_path}")
    if not manifest_path.exists():
        fail(f"刪除前缺少 manifest: {manifest_path}")
    validate_jsonl(output_path)
    validate_coverage(output_path, total_pages=total_pages, allow_skipped=False)
    validate_manifest(output_path, require_originals=keep_pdf or keep_pages, total_pages=total_pages)
    if not keep_pdf and not keep_pages:
        fail("禁止同時刪除 PDF 與頁圖；至少保留一種 source-of-truth")
    manifest = load_progress(manifest_path)
    pdf_exists = Path(manifest.get("pdf_path", "")).exists() if manifest else False
    page_paths = [Path(p.get("png_path", "")) for p in (manifest or {}).get("pages", [])]
    pages_exist = bool(page_paths) and all(p.exists() for p in page_paths)
    can_delete_pdf = (not keep_pdf) and keep_pages and pages_exist
    if not keep_pdf and not pages_exist:
        fail("想刪 PDF 時必須保留完整 pages PNG 證據")
    report = {
        "schema_version": SCHEMA_VERSION,
        "can_delete_pdf": can_delete_pdf,
        "keep_pdf": keep_pdf,
        "keep_pages": keep_pages,
        "pdf_exists": pdf_exists,
        "pages_exist": pages_exist,
        "output_path": str(output_path),
        "jsonl_sha256": sha256_file(output_path),
        "manifest_sha256": sha256_file(manifest_path),
        "final_report_sha256": sha256_file(final_report_path),
        "warning": "can_delete_pdf=true 只代表 page-raster OCR 證據已封存，不代表原始 PDF 可被重建或完整審計。",
    }
    gate_path = Path(str(output_path) + ".deletion-report.json")
    atomic_write_json(gate_path, report, overwrite=overwrite)
    print(f"OK:deletion-gate:{gate_path}:can_delete_pdf={can_delete_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="pdf-ocr hard-rule guard")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cmd = sub.add_parser("check-command")
    p_cmd.add_argument("command", nargs=argparse.REMAINDER)

    p_existing = sub.add_parser("check-existing-jsonl")
    p_existing.add_argument("path", type=Path)
    p_existing.add_argument("--allow-missing-source", action="store_true")

    p_validate = sub.add_parser("validate-jsonl")
    p_validate.add_argument("path", type=Path)

    p_single = sub.add_parser("check-single-book")
    p_single.add_argument("--current-output", type=Path, default=None)

    p_progress = sub.add_parser("validate-progress")
    p_progress.add_argument("path", type=Path)

    p_manifest = sub.add_parser("write-manifest")
    p_manifest.add_argument("pdf_path", type=Path)
    p_manifest.add_argument("output_path", type=Path)
    p_manifest.add_argument("--png-dir", type=Path, default=None)
    p_manifest.add_argument("--total-pages", type=int, default=None)
    p_manifest.add_argument("--overwrite", action="store_true")

    p_validate_manifest = sub.add_parser("validate-manifest")
    p_validate_manifest.add_argument("output_path", type=Path)
    p_validate_manifest.add_argument("--total-pages", type=int, default=None)
    p_validate_manifest.add_argument("--allow-missing-originals", action="store_true")

    p_coverage = sub.add_parser("validate-coverage")
    p_coverage.add_argument("output_path", type=Path)
    p_coverage.add_argument("--total-pages", type=int, required=True)
    p_coverage.add_argument("--allow-skipped", action="store_true")

    p_final = sub.add_parser("finalize-audit")
    p_final.add_argument("output_path", type=Path)
    p_final.add_argument("--total-pages", type=int, required=True)
    p_final.add_argument("--overwrite", action="store_true")

    p_delete = sub.add_parser("deletion-gate")
    p_delete.add_argument("output_path", type=Path)
    p_delete.add_argument("--total-pages", type=int, required=True)
    p_delete.add_argument("--keep-pdf", action="store_true")
    p_delete.add_argument("--keep-pages", action="store_true")
    p_delete.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.cmd == "check-command":
        check_command(" ".join(args.command))
    elif args.cmd == "check-existing-jsonl":
        validate_existing_jsonl(args.path, strict_missing=not args.allow_missing_source)
    elif args.cmd == "validate-jsonl":
        validate_jsonl(args.path)
    elif args.cmd == "check-single-book":
        check_single_book(args.current_output)
    elif args.cmd == "validate-progress":
        validate_progress(args.path)
    elif args.cmd == "write-manifest":
        write_manifest(args.pdf_path, args.output_path, args.png_dir, args.total_pages, args.overwrite)
    elif args.cmd == "validate-manifest":
        validate_manifest(args.output_path, require_originals=not args.allow_missing_originals, total_pages=args.total_pages)
    elif args.cmd == "validate-coverage":
        validate_coverage(args.output_path, args.total_pages, args.allow_skipped)
    elif args.cmd == "finalize-audit":
        finalize_audit(args.output_path, args.total_pages, args.overwrite)
    elif args.cmd == "deletion-gate":
        deletion_gate(args.output_path, args.total_pages, args.keep_pdf, args.keep_pages, args.overwrite)


if __name__ == "__main__":
    main()
