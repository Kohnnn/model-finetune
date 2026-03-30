from __future__ import annotations

from pathlib import Path

from ocr_pipeline.process_pdfs import (
    build_document_metadata,
    discover_files,
    is_boilerplate_chunk,
    strip_vcsc_disclaimers,
    trim_tail_sections,
)


def test_discover_files_skips_office_lock_files(tmp_path: Path) -> None:
    keep = tmp_path / "report.docx"
    skip = tmp_path / "~$report.docx"
    keep.write_text("ok", encoding="utf-8")
    skip.write_text("lock", encoding="utf-8")

    files = discover_files(tmp_path, [".docx"])

    assert files == [keep]


def test_build_document_metadata_adds_retrieval_fields(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_dataset"
    source = (
        input_dir
        / "Strategy"
        / "Strategy 2026"
        / "[VN] VietnamStrategy2026-Transportation.pptx"
    )
    source.parent.mkdir(parents=True)
    source.write_text("placeholder", encoding="utf-8")

    metadata = build_document_metadata(source, input_dir, "Doanh thu va loi nhuan tang")

    assert metadata["relative_source"].endswith(
        "[VN] VietnamStrategy2026-Transportation.pptx"
    )
    assert metadata["doc_id"].startswith("strategy_strategy_2026")
    assert metadata["title"] == "[VN] VietnamStrategy2026 Transportation"
    assert metadata["year"] == 2026
    assert metadata["language"] == "vi"
    assert metadata["file_extension"] == ".pptx"


def test_strip_vcsc_disclaimers_removes_english_tail_markers() -> None:
    text = (
        "Revenue growth improved meaningfully in the quarter. "
        "Analyst Certification I hereby certify that the views expressed in this report..."
    )

    cleaned = strip_vcsc_disclaimers(text)

    assert cleaned == "Revenue growth improved meaningfully in the quarter."


def test_trim_tail_sections_drops_contact_page_before_generic_trim() -> None:
    pages = [
        "Core earnings improved with better fee income and lower funding costs.",
        "Contacts\nFor investment advice, trade execution or other enquiries, clients should contact their local sales representative.",
    ]

    trimmed = trim_tail_sections(pages, trim_tail_pages=0)

    assert trimmed == [pages[0]]


def test_is_boilerplate_chunk_detects_disclaimer_text() -> None:
    chunk = (
        "For investment advice, trade execution or other enquiries, clients should contact "
        "their local sales representative. Disclaimer Analyst Certification of Independence ..."
    )

    assert is_boilerplate_chunk(chunk) is True
