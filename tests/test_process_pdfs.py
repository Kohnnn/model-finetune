from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from ocr_pipeline.process_pdfs import (
    add_near_duplicate_family,
    build_document_metadata,
    canonicalize_document_families,
    chunk_sections,
    discover_files,
    extract_docx_pages,
    is_boilerplate_chunk,
    parse_args,
    process_dataset,
    strip_vcsc_disclaimers,
    trim_tail_sections,
)
from ocr_pipeline.export_markdown_reports import (
    clean_line,
    content_fingerprint,
    infer_visual_kind,
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
    assert "source" not in metadata
    assert metadata["doc_id"].startswith("strategy_strategy_2026")
    assert metadata["title"] == "[VN] VietnamStrategy2026 Transportation"
    assert metadata["year"] == 2026
    assert metadata["language"] == "vi"
    assert metadata["file_extension"] == ".pptx"
    assert len(metadata["source_file_sha256"]) == 64
    assert len(metadata["content_sha256"]) == 64
    assert metadata["document_family_id"] == f"family_{metadata['content_sha256']}"
    assert metadata["extraction_method"] == "primary"
    assert metadata["parser_schema_version"] == "2"


def test_document_family_id_is_deterministic_for_normalized_content(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_dataset"
    first = input_dir / "first.pdf"
    second = input_dir / "second.pdf"
    input_dir.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    first_metadata = build_document_metadata(first, input_dir, "Revenue  growth improved.")
    second_metadata = build_document_metadata(second, input_dir, " revenue growth IMPROVED. ")

    assert first_metadata["document_family_id"] == second_metadata["document_family_id"]


def test_near_duplicate_families_merge_transitive_chain() -> None:
    families: list[tuple[int, str]] = []
    parents: dict[str, str] = {}
    assert add_near_duplicate_family(0b0000, "family-a", families, parents, 1) is False
    assert add_near_duplicate_family(0b0001, "family-b", families, parents, 1) is True
    assert add_near_duplicate_family(0b0011, "family-c", families, parents, 1) is True
    rows = [
        {"metadata": {"document_family_id": "family-a"}},
        {"metadata": {"document_family_id": "family-b"}},
        {"metadata": {"document_family_id": "family-c"}},
    ]

    canonicalize_document_families(rows, parents)

    assert {row["metadata"]["document_family_id"] for row in rows} == {"family-a"}


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


def test_parse_args_defaults_to_safe_tail_preservation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["process_pdfs.py"])

    assert parse_args().trim_tail_pages == 0


def test_is_boilerplate_chunk_detects_disclaimer_text() -> None:
    chunk = (
        "For investment advice, trade execution or other enquiries, clients should contact "
        "their local sales representative. Disclaimer Analyst Certification of Independence ..."
    )

    assert is_boilerplate_chunk(chunk) is True


def test_chunk_sections_preserves_sentence_and_page_provenance() -> None:
    chunks = chunk_sections(
        [(4, "One two three. Four five six."), (7, "Seven eight nine.")],
        chunk_words=6,
        overlap_words=3,
        min_chunk_words=3,
    )

    assert chunks[0]["text"] == "One two three. Four five six."
    assert chunks[0]["start_page"] == 4
    assert chunks[1]["source_page_numbers"] == "[4, 7]"
    assert chunks[1]["source_word_start"] == 3
    assert chunks[1]["source_word_end"] == 9


def test_extract_docx_pages_keeps_table_in_document_order(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    source = tmp_path / "report.docx"
    document = docx.Document()
    document.add_paragraph("Opening analysis")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    document.add_paragraph("Closing analysis")
    document.save(source)

    extracted = "\n".join(extract_docx_pages(source))

    assert extracted.index("Opening analysis") < extracted.index("Metric | Value") < extracted.index("Closing analysis")


def test_process_dataset_rejects_default_output_pilot_overwrite() -> None:
    args = argparse.Namespace(
        limit=1,
        output_dir=Path(__file__).resolve().parents[1] / "ocr_pipeline",
        allow_pilot_overwrite=False,
    )

    with pytest.raises(RuntimeError, match="allow-pilot-overwrite"):
        process_dataset(args)


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


def test_infer_visual_kind_does_not_treat_bare_f_as_forecast() -> None:
    assert infer_visual_kind("Factory footprint by province") == "figure/chart"
    assert infer_visual_kind("2026 forecast earnings") == "forecast chart/table"
    assert infer_visual_kind("2026F earnings") == "forecast chart/table"
