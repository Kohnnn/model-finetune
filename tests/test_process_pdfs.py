from __future__ import annotations

from pathlib import Path

from ocr_pipeline.process_pdfs import (
    build_document_metadata,
    discover_files,
    is_boilerplate_chunk,
    strip_vcsc_disclaimers,
    trim_tail_sections,
)
from ocr_pipeline.export_markdown_reports import (
    clean_line,
    content_fingerprint,
    page_is_compliance,
    pages_to_markdown,
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


def test_pages_to_markdown_keeps_structure_and_removes_disclaimer() -> None:
    pages = [
        "Investment thesis\nRevenue growth improved and margin pressure eased.",
        "Analyst Certification\nThis report is provided, for information purposes only.",
    ]

    markdown, body_words = pages_to_markdown(
        pages,
        title="ACB Update",
        relative_source="Banking/ACB.pdf",
        min_page_words=1,
    )

    assert markdown.startswith("# ACB Update")
    assert "## Document Metadata" in markdown
    assert "## Page 1" in markdown
    assert "### Investment thesis" in markdown
    assert "Revenue growth improved" in markdown
    assert "Analyst Certification" not in markdown
    assert body_words > 0


def test_clean_line_drops_noise_keeps_analysis() -> None:
    assert clean_line("See important disclosures at the end of this report") == ""
    assert clean_line("Analyst Certification") == ""
    assert clean_line("Tel: +84 28 3914 3588") == ""
    assert clean_line("research@vietcap.com.vn") == ""
    assert clean_line("123") == ""
    kept = clean_line("HPG target price raised to VND 33,500 on margin expansion")
    assert kept == "HPG target price raised to VND 33,500 on margin expansion"


def test_page_is_compliance_detects_boilerplate_pages() -> None:
    compliance = (
        "Analyst Certification. Important disclosures. "
        "Rating system: Buy means total return above 15%."
    )
    assert page_is_compliance(compliance) is True
    analytical = "HPG margins expanded on lower coke costs and higher ASPs in 2Q."
    assert page_is_compliance(analytical) is False


def test_content_fingerprint_ignores_metadata_header() -> None:
    base_body = "## Report Body\n\n## Page 1\n\nHPG margins expanded on lower costs."
    doc_a = "# HPG Report A\n\n## Document Metadata\n\n- Source: `a.pdf`\n" + base_body
    doc_b = "# HPG Report B\n\n## Document Metadata\n\n- Source: `b.pptx`\n" + base_body
    doc_c = "# HPG Report C\n\n## Document Metadata\n\n- Source: `c.pdf`\n" + (
        "## Report Body\n\n## Page 1\n\nVCB net interest margin compressed in 3Q."
    )
    assert content_fingerprint(doc_a) == content_fingerprint(doc_b)
    assert content_fingerprint(doc_a) != content_fingerprint(doc_c)
