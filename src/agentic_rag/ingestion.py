"""Independent, deterministic PDF extraction and traceable chunking."""

from __future__ import annotations

import hashlib
import json
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import ContractError
from .validation import validate_dataset


@dataclass(frozen=True, slots=True)
class PageSpec:
    page: int
    chapter: int
    chapter_title: str


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    document_id: str
    source_url: str
    source_sha256: str
    license: str
    license_url: str
    version: str
    language: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    pages: tuple[PageSpec, ...]


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"corpus config {key}: expected a non-empty string")
    return value.strip()


def load_corpus_config(path: Path) -> CorpusConfig:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid corpus config: {error}") from error
    if not isinstance(row, dict):
        raise ContractError(f"{path}: expected a JSON object")
    size = row.get("chunk_size_chars")
    overlap = row.get("chunk_overlap_chars")
    if type(size) is not int or size < 100:
        raise ContractError("corpus config chunk_size_chars: expected integer >= 100")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ContractError("corpus config chunk_overlap_chars: expected 0 <= overlap < size")
    ranges = row.get("page_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ContractError("corpus config page_ranges: expected a non-empty array")
    pages: list[PageSpec] = []
    for index, value in enumerate(ranges):
        if not isinstance(value, dict):
            raise ContractError(f"corpus config page_ranges[{index}]: expected an object")
        start, end, chapter = value.get("start"), value.get("end"), value.get("chapter")
        title = value.get("chapter_title")
        if (
            type(start) is not int
            or type(end) is not int
            or type(chapter) is not int
            or start < 1
            or end < start
            or not isinstance(title, str)
            or not title.strip()
        ):
            raise ContractError(f"corpus config page_ranges[{index}]: invalid page/chapter data")
        pages.extend(PageSpec(page, chapter, title.strip()) for page in range(start, end + 1))
    page_numbers = [spec.page for spec in pages]
    if len(set(page_numbers)) != len(page_numbers):
        raise ContractError("corpus config page_ranges: physical pages must not overlap")
    if len(pages) < 30:
        raise ContractError(f"corpus config: requires at least 30 pages, found {len(pages)}")
    return CorpusConfig(
        document_id=_required_text(row, "document_id"),
        source_url=_required_text(row, "source_url"),
        source_sha256=_required_text(row, "source_sha256"),
        license=_required_text(row, "license"),
        license_url=_required_text(row, "license_url"),
        version=_required_text(row, "version"),
        language=_required_text(row, "language"),
        chunk_size_chars=size,
        chunk_overlap_chars=overlap,
        pages=tuple(pages),
    )


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractError(f"{path}: could not read PDF: {error}") from error


def extract_pdf_page(pdf_path: Path, page: int) -> str:
    command = [
        "pdftotext",
        "-layout",
        "-f",
        str(page),
        "-l",
        str(page),
        str(pdf_path),
        "-",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        raise ContractError(f"pdftotext is unavailable: {error}") from error
    if result.returncode:
        message = result.stderr.strip() or "unknown pdftotext error"
        raise ContractError(f"PDF page {page}: extraction failed: {message}")
    normalized = unicodedata.normalize("NFC", result.stdout.replace("\f", ""))
    lines = [line.rstrip() for line in normalized.splitlines()]
    text = "\n".join(lines).strip("\n")
    if not text.strip():
        raise ContractError(f"PDF page {page}: extracted text is empty")
    return text


def fixed_window_chunks(text: str, *, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Split exact source text into overlapping, gap-free character windows."""
    if not text or size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("text/size/overlap do not define a valid chunk window")
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end)
            )
            if boundary > start:
                end = boundary + 1
        chunks.append((start, end, text[start:end]))
        if end == len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            raise AssertionError("chunk window did not make progress")
        start = next_start
    return chunks


def prepare_corpus(
    pdf_path: Path,
    config_path: Path,
    output_directory: Path,
    *,
    extractor: Callable[[Path, int], str] = extract_pdf_page,
) -> dict[str, Any]:
    config = load_corpus_config(config_path)
    actual_sha256 = _sha256(pdf_path)
    if actual_sha256 != config.source_sha256:
        raise ContractError(
            f"PDF checksum mismatch: expected {config.source_sha256}, found {actual_sha256}"
        )
    pages_path = output_directory / "pages.json"
    chunks_path = output_directory / "chunks.json"
    stats_path = output_directory / "ingestion_stats.json"
    existing = [path for path in (pages_path, chunks_path, stats_path) if path.exists()]
    if existing:
        raise ContractError(
            "refusing to overwrite corpus output: " + ", ".join(str(path) for path in existing)
        )

    pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for spec in config.pages:
        text = extractor(pdf_path, spec.page)
        page_id = f"{config.document_id}:pdf-{spec.page:03d}"
        page_metadata = {
            "page_id": page_id,
            "chapter": spec.chapter,
            "chapter_title": spec.chapter_title,
            "language": config.language,
            "version": config.version,
            "source_sha256": config.source_sha256,
        }
        pages.append(
            {
                "id": page_id,
                "source_id": config.document_id,
                "source": config.source_url,
                "page": spec.page,
                "text": text,
                "metadata": page_metadata,
            }
        )
        for start, end, chunk_text in fixed_window_chunks(
            text, size=config.chunk_size_chars, overlap=config.chunk_overlap_chars
        ):
            identity = f"{config.document_id}:{spec.page}:{start}:{end}:{chunk_text}".encode()
            chunks.append(
                {
                    "id": hashlib.sha256(identity).hexdigest(),
                    "source_id": config.document_id,
                    "source": config.source_url,
                    "page": spec.page,
                    "text": chunk_text,
                    "metadata": {
                        **page_metadata,
                        "start_char": start,
                        "end_char": end,
                        "chunking": "fixed_window",
                        "chunk_size_chars": config.chunk_size_chars,
                        "chunk_overlap_chars": config.chunk_overlap_chars,
                    },
                }
            )
    validate_dataset(
        chunks,
        [
            {
                "id": "placeholder",
                "split": "dev",
                "question": "?",
                "answerable": False,
                "expected_behavior": "search_and_abstain",
                "absence_reason": "Structural chunk validation only.",
            }
        ],
        split="dev",
    )
    stats: dict[str, Any] = {
        "document_id": config.document_id,
        "source": config.source_url,
        "source_sha256": config.source_sha256,
        "license": config.license,
        "license_url": config.license_url,
        "pages": len(pages),
        "chunks": len(chunks),
        "total_characters": sum(len(page["text"]) for page in pages),
        "chunk_size_chars": config.chunk_size_chars,
        "chunk_overlap_chars": config.chunk_overlap_chars,
        "extraction_backend": "pdftotext-layout",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    for destination, payload in (
        (pages_path, pages),
        (chunks_path, chunks),
        (stats_path, stats),
    ):
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return stats
