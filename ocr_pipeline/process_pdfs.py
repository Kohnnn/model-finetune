from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Iterable

LOGGER = logging.getLogger(__name__)

VIETNAMESE_CHAR_PATTERN = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")

TAIL_SECTION_MARKERS = [
    "analyst certification",
    "analyst certification of independence",
    "important disclosures",
    "vCSC rating system",
    "valuation methodology",
    "for investment advice, trade execution or other enquiries",
    "for investment advice",
    "this report is provided, for information purposes only",
    "major us institutional investors",
    "u.k. and european economic area",
    "hong kong:",
    "new zealand:",
    "contacts",
    "contact us",
    "xác nhận của chuyên viên phân tích",
    "phương pháp định giá và hệ thống khuyến nghị của vcsc",
    "phương pháp định giá: để xác định giá mục tiêu",
    "phòng giao dịch chứng khoán",
    "phòng nghiên cứu và phân tích",
    "báo cáo này được viết và phát hành bởi",
    "công ty cổ phần chứng khoán bản việt",
]

CONTACT_PAGE_MARKERS = [
    "contacts",
    "contact us",
    "local sales representative",
    "for investment advice, trade execution or other enquiries",
    "decker&co",
]

HEAD_SECTION_MARKERS = [
    "update report",
    "see important disclosure",
    "www.vietcap.com.vn",
    "vietcap securities",
    "disclaimer",
]

NOISE_PATTERNS = [
    r"(?i)^\s*figure\s*\d+.*?source:\s*",
    r"(?i)^\s*bảng\s*\d+.*?nguồn:\s*",
    r"(?i)^\s*table\s*\d+.*?source:\s*",
    r"(?i)^\s*page\s*\d+\s+of\s+\d+",
    r"(?i)^\s*\d{1,3}\.\s+[a-z\-\s]+$",
    r"(?i)^\s*mục\s*lục\s*c",
    r"(?i)^\s*tài\s*liệu\s*tham\s*khảo",
    r"(?i)nguồn:\s*\w+",
    r"(?i)ngừng\s*theo\s*dõi",
    r"(?i)báo\s*cáo\s*tài\s*chính\s*nguồn:",
    r"(?i)^\s*[+\-]?\$?[\d,\.]+\s*%?$",
    r"(?i)^\s*[+\-]?\d+\.?\d*%\s*$",
    r"(?i)(buy|sell|outperform|underperform|neutral)\s*=\s*",
    r"(?i)vietcap rating system",
    r"(?i)^\+?\d{6,}",
    r"(?i)^\s*[ 	]+$",
]

_NOISE_PATTERN_RE = [re.compile(p) for p in NOISE_PATTERNS]

ANALYTICAL_MARKERS = [
    "target price",
    "valuation",
    "recommendation",
    "earnings",
    "margin",
    "profit",
    "forecast",
    "upside",
    "downside",
    "khuyến nghị",
    "giá mục tiêu",
    "định giá",
    "lợi nhuận",
    "biên lợi nhuận",
    "dự báo",
]

SYSTEM_PROMPT = (
    "You are a senior equity research analyst. Answer queries using a highly "
    "professional financial research tone, voice, and style, based strictly "
    "on the provided context. Focus on analytical synthesis and strategic "
    "insights rather than listing facts."
)

USER_TASK_PROMPT = (
    "Task: Deliver expert equity research commentary and strategic evaluation "
    "based on the context above. Prioritize deep analysis over factual reporting."
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Parse dataset documents into RAG chunks and finetuning templates "
            "(JSONL output)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "raw_dataset",
        help="Root directory containing source files (default: raw_dataset).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory for chroma_chunks.jsonl and finetune_template.jsonl "
            "(default: ocr_pipeline)."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf", ".docx", ".pptx"],
        help=(
            "File extensions to include, e.g. .pdf .docx .pptx "
            "(default: .pdf .docx .pptx)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files processed (useful for pilot runs).",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["head", "random"],
        default="random",
        help="How files are selected when --limit is set (default: random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic pilot sampling (default: 3407).",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=800,
        help="Chunk size in words (default: 800).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=100,
        help="Word overlap between chunks (default: 100).",
    )
    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=200,
        help="Discard chunks smaller than this threshold (default: 200).",
    )
    parser.add_argument(
        "--trim-tail-pages",
        type=int,
        default=3,
        help=(
            "Drop this many trailing pages/slides/sections as likely disclaimers "
            "when possible (default: 3)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def normalize_extensions(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        ext = value.strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return sorted(set(normalized))


def discover_files(input_dir: Path, extensions: list[str]) -> list[Path]:
    files: list[Path] = []
    for extension in extensions:
        files.extend(input_dir.rglob(f"*{extension}"))
    return sorted(
        (p for p in files if p.is_file() and not should_skip_file(p)),
        key=lambda p: str(p).lower(),
    )


def should_skip_file(file_path: Path) -> bool:
    return file_path.name.startswith("~$")


def select_files(
    files: list[Path],
    limit: int | None,
    sample_mode: str,
    seed: int,
) -> list[Path]:
    if limit is None or limit >= len(files):
        return files
    if sample_mode == "head":
        return files[:limit]
    rng = random.Random(seed)
    return sorted(rng.sample(files, limit), key=lambda p: str(p).lower())


def normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text)
    return compact.strip()


def normalize_for_matching(text: str) -> str:
    return normalize_text(text).casefold()


def count_marker_hits(text: str, markers: list[str]) -> int:
    normalized = normalize_for_matching(text)
    return sum(normalize_for_matching(marker) in normalized for marker in markers)


def strip_head_boilerplate(text: str) -> str:
    text_lower = normalize_for_matching(text)
    found_marker = None
    found_idx = len(text)
    for marker in HEAD_SECTION_MARKERS:
        norm_marker = normalize_for_matching(marker)
        idx = text_lower.find(norm_marker)
        if idx != -1 and idx < found_idx:
            found_idx = idx
            found_marker = marker
    if found_marker is None:
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_lower = normalize_for_matching(line)
        if normalize_for_matching(found_marker) in line_lower:
            return "\n".join(lines[i + 1 :]).strip()
    return text


def matches_noise_pattern(text: str) -> bool:
    for pattern_re in _NOISE_PATTERN_RE:
        if pattern_re.search(text):
            return True
    return False


def has_excessive_numbers(text: str) -> bool:
    if not text:
        return False
    digits = sum(c.isdigit() for c in text)
    return (digits / len(text)) > 0.30


def count_analytical_markers(text: str) -> int:
    return count_marker_hits(text, ANALYTICAL_MARKERS)


def is_boilerplate_page(text: str) -> bool:
    word_count = len(text.split())
    tail_hits = count_marker_hits(text, TAIL_SECTION_MARKERS)
    contact_hits = count_marker_hits(text, CONTACT_PAGE_MARKERS)
    analytical_hits = count_marker_hits(text, ANALYTICAL_MARKERS)
    normalized = normalize_for_matching(text)

    if not normalized:
        return False
    if contact_hits >= 1 and word_count <= 220:
        return True
    if (
        "analyst certification" in normalized
        or "xác nhận của chuyên viên phân tích" in normalized
    ):
        return True
    if tail_hits >= 2 and analytical_hits == 0:
        return True
    if tail_hits >= 3:
        return True
    return False


def strip_vcsc_disclaimers(text: str) -> str:
    """Truncates disclaimer and contact boilerplate typically found at report tails."""

    text_lower = normalize_for_matching(text)
    earliest_idx = len(text)

    for marker in TAIL_SECTION_MARKERS:
        norm_marker = normalize_for_matching(marker)
        idx = text_lower.find(norm_marker)
        if idx != -1 and idx < earliest_idx:
            earliest_idx = idx

    if earliest_idx < len(text):
        return text[:earliest_idx].strip()
    return text


def infer_document_title(file_path: Path) -> str:
    title = re.sub(r"[_\-]+", " ", file_path.stem)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def infer_document_year(relative_source: str) -> int | None:
    matches = YEAR_PATTERN.findall(relative_source)
    if not matches:
        return None
    return int(matches[0])


def infer_document_language(relative_source: str, text: str) -> str:
    normalized_source = relative_source.lower()
    if "[vn]" in normalized_source or "vietnamese" in normalized_source:
        return "vi"
    if VIETNAMESE_CHAR_PATTERN.search(text):
        return "vi"
    return "en"


def build_doc_id(relative_source: str) -> str:
    normalized_source = relative_source.lower()
    doc_id = re.sub(r"[^a-z0-9]+", "_", normalized_source).strip("_")
    digest = hashlib.md5(normalized_source.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_{digest}"


def build_document_metadata(file_path: Path, input_dir: Path, text: str) -> dict:
    relative_source = str(file_path.relative_to(input_dir))
    return {
        "source": str(file_path),
        "relative_source": relative_source,
        "doc_id": build_doc_id(relative_source),
        "title": infer_document_title(file_path),
        "year": infer_document_year(relative_source),
        "language": infer_document_language(relative_source, text),
        "file_extension": file_path.suffix.lower(),
    }


def extract_pdf_pages(file_path: Path) -> list[str]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'PyMuPDF'. Install with: pip install PyMuPDF"
        ) from exc

    pages: list[str] = []
    with fitz.open(file_path) as pdf_doc:
        for page in pdf_doc:
            pages.append(page.get_text("text"))
    return pages


def extract_docx_pages(file_path: Path) -> list[str]:
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'python-docx'. Install with: pip install python-docx"
        ) from exc

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    grouped_pages: list[str] = []
    for i in range(0, len(paragraphs), 30):
        grouped_pages.append("\n".join(paragraphs[i : i + 30]))
    return grouped_pages


def extract_pptx_pages(file_path: Path) -> list[str]:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'python-pptx'. Install with: pip install python-pptx"
        ) from exc

    presentation = Presentation(file_path)
    slides: list[str] = []
    for slide in presentation.slides:
        parts: list[str] = []
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if text and text.strip():
                parts.append(text)
        slides.append("\n".join(parts))
    return slides


def extract_pages(file_path: Path) -> list[str]:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        pages = extract_pdf_pages(file_path)
    elif extension == ".docx":
        pages = extract_docx_pages(file_path)
    elif extension == ".pptx":
        pages = extract_pptx_pages(file_path)
    else:
        raise ValueError(f"Unsupported extension: {extension}")

    normalized_pages: list[str] = []
    for page in pages:
        normalized = normalize_text(page)
        if normalized:
            normalized_pages.append(normalized)
    return normalized_pages


def trim_tail_sections(pages: list[str], trim_tail_pages: int) -> list[str]:
    if not pages:
        return []

    trimmed_pages = list(pages)
    while len(trimmed_pages) > 1 and is_boilerplate_page(trimmed_pages[-1]):
        trimmed_pages.pop()

    if trim_tail_pages <= 0:
        return trimmed_pages
    if len(trimmed_pages) > trim_tail_pages:
        return trimmed_pages[:-trim_tail_pages]
    if len(trimmed_pages) > 1:
        return trimmed_pages[:-1]
    return trimmed_pages


def is_boilerplate_chunk(text: str) -> bool:
    normalized = normalize_for_matching(text)
    if not normalized:
        return True

    word_count = len(text.split())
    tail_hits = count_marker_hits(text, TAIL_SECTION_MARKERS)
    contact_hits = count_marker_hits(text, CONTACT_PAGE_MARKERS)
    analytical_hits = count_analytical_markers(text)

    if contact_hits >= 1 and word_count <= 220:
        return True
    if (
        "analyst certification" in normalized
        or "xác nhận của chuyên viên phân tích" in normalized
    ):
        return True
    if tail_hits >= 2 and analytical_hits == 0:
        return True
    if tail_hits >= 3:
        return True
    if matches_noise_pattern(text):
        return True
    if has_excessive_numbers(text):
        return True
    return False


def is_quality_chunk(text: str) -> bool:
    if is_boilerplate_chunk(text):
        return False
    word_count = len(text.split())
    if word_count < 200:
        return False
    analytical_count = count_analytical_markers(text)
    if analytical_count < 2:
        return False
    return True


def chunk_text(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[str]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if len(chunk) >= min_chunk_words:
            chunks.append(" ".join(chunk))
    return chunks


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_dataset(args: argparse.Namespace) -> None:
    start_time = perf_counter()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    extensions = normalize_extensions(args.extensions)
    files = discover_files(args.input_dir, extensions)
    selected_files = select_files(files, args.limit, args.sample_mode, args.seed)

    LOGGER.info("Discovered %d files (extensions=%s)", len(files), ",".join(extensions))
    LOGGER.info("Selected %d files for this run", len(selected_files))

    if not selected_files:
        raise RuntimeError(
            "No files matched the given input directory and extension filters"
        )

    chroma_chunks: list[dict] = []
    finetune_templates: list[dict] = []
    failed_files: list[str] = []

    for index, file_path in enumerate(selected_files, start=1):
        file_start = perf_counter()
        try:
            pages = extract_pages(file_path)
            pages = trim_tail_sections(pages, args.trim_tail_pages)
            text = "\n\n".join(pages)
            text = strip_vcsc_disclaimers(text)
            document_metadata = build_document_metadata(file_path, args.input_dir, text)
            text = strip_head_boilerplate(text)
            chunks = chunk_text(
                text=text,
                chunk_words=args.chunk_words,
                overlap_words=args.overlap_words,
                min_chunk_words=args.min_chunk_words,
            )
            chunks = [chunk for chunk in chunks if is_quality_chunk(chunk)]

            if not chunks:
                LOGGER.warning(
                    "[%d/%d] No valid chunks: %s", index, len(selected_files), file_path
                )
                continue

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{document_metadata['doc_id']}_chunk_{chunk_index:04d}"

                chroma_chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            **document_metadata,
                            "chunk_index": chunk_index,
                            "chunk_word_count": len(chunk.split()),
                        },
                    }
                )

                finetune_templates.append(
                    {
                        "metadata": {
                            **document_metadata,
                            "chunk_index": chunk_index,
                            "chunk_word_count": len(chunk.split()),
                        },
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"Context:\n{chunk}\n\n{USER_TASK_PROMPT}",
                            },
                            {
                                "role": "assistant",
                                "content": "",
                            },
                        ],
                    }
                )

            elapsed = perf_counter() - file_start
            LOGGER.info(
                "[%d/%d] Parsed %s -> %d chunks in %.2fs",
                index,
                len(selected_files),
                file_path.name,
                len(chunks),
                elapsed,
            )
        except Exception as exc:  # noqa: PERF203
            failed_files.append(str(file_path))
            LOGGER.error(
                "[%d/%d] Failed %s: %s", index, len(selected_files), file_path, exc
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chroma_path = args.output_dir / "chroma_chunks.jsonl"
    finetune_path = args.output_dir / "finetune_template.jsonl"

    write_jsonl(chroma_path, chroma_chunks)
    write_jsonl(finetune_path, finetune_templates)

    elapsed_total = perf_counter() - start_time
    processed_files = len(selected_files) - len(failed_files)
    avg_chunks_per_file = (
        (len(chroma_chunks) / processed_files) if processed_files else 0.0
    )

    LOGGER.info("Run complete in %.2fs", elapsed_total)
    LOGGER.info(
        "Processed files: %d | Failed files: %d", processed_files, len(failed_files)
    )
    LOGGER.info(
        "Total chunks: %d | Avg chunks/file: %.2f",
        len(chroma_chunks),
        avg_chunks_per_file,
    )
    LOGGER.info("Output: %s", chroma_path)
    LOGGER.info("Output: %s", finetune_path)

    if failed_files:
        failed_log = args.output_dir / "parse_failures.log"
        failed_log.write_text("\n".join(failed_files), encoding="utf-8")
        LOGGER.warning("Failed file list written to %s", failed_log)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        process_dataset(args)
    except Exception as exc:  # noqa: PERF203
        LOGGER.error("Parsing failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
