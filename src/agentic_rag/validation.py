"""Dataset loading and cross-record validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .domain import BenchmarkQuestion, Chunk, ContractError

Split = Literal["dev", "test", "all"]


@dataclass(frozen=True, slots=True)
class SplitCounts:
    answerable: int = 0
    unanswerable: int = 0


@dataclass(frozen=True, slots=True)
class DatasetReport:
    chunks: int
    questions: int
    dev: SplitCounts
    test: SplitCounts

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-ready report without exposing benchmark gold."""
        return asdict(self)


def load_json_array(path: Path) -> list[object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ContractError(f"{path}: could not read file: {error}") from error
    except json.JSONDecodeError as error:
        raise ContractError(
            f"{path}:{error.lineno}:{error.colno}: invalid JSON: {error.msg}"
        ) from error
    if not isinstance(value, list):
        raise ContractError(f"{path}: expected a top-level JSON array")
    return value


def _unique_ids(rows: list[Chunk] | list[BenchmarkQuestion], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            raise ContractError(f"{label}: duplicate id {row.id!r}")
        seen.add(row.id)


def _gold_resolves(question: BenchmarkQuestion, chunks: list[Chunk]) -> None:
    for index, evidence in enumerate(question.gold):
        matching_page = [
            chunk
            for chunk in chunks
            if chunk.source_id == evidence.source_id and chunk.page == evidence.page
        ]
        single_chunk = any(
            chunk.source_id == evidence.source_id
            and chunk.page == evidence.page
            and evidence.text in chunk.text
            for chunk in chunks
        )
        if not single_chunk and evidence.text not in _reconstruct_page(matching_page):
            raise ContractError(
                f"question {question.id!r} gold[{index}]: evidence does not resolve to "
                "a single chunk or reconstructed page with the same source_id and page"
            )


def _reconstruct_page(chunks: list[Chunk]) -> str:
    """Rebuild source text from exact Day 6 character spans, preserving gaps."""
    spans: list[tuple[int, int, str]] = []
    for chunk in chunks:
        start = chunk.metadata.get("start_char")
        end = chunk.metadata.get("end_char")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or len(chunk.text) != end - start
        ):
            continue
        spans.append((start, end, chunk.text))
    if not spans:
        return ""
    page = ["\0"] * max(end for _, end, _ in spans)
    for start, end, text in spans:
        for offset, character in enumerate(text, start=start):
            if page[offset] not in {"\0", character}:
                raise ContractError(
                    f"chunks: conflicting source text at character {offset} during reconstruction"
                )
            page[offset] = character
        if offset + 1 != end:
            raise AssertionError("validated span length did not reach end_char")
    return "".join(page)


def _absence_terms_do_not_resolve(question: BenchmarkQuestion, chunks: list[Chunk]) -> None:
    for term in question.absence_terms:
        matching = [chunk.id for chunk in chunks if term.casefold() in chunk.text.casefold()]
        if matching:
            preview = ", ".join(matching[:3])
            raise ContractError(
                f"question {question.id!r} absence term {term!r} appears in chunk(s): {preview}"
            )


def _counts(questions: list[BenchmarkQuestion], split: str) -> SplitCounts:
    selected = [question for question in questions if question.split == split]
    return SplitCounts(
        answerable=sum(question.answerable for question in selected),
        unanswerable=sum(not question.answerable for question in selected),
    )


def _require_minimums(report: DatasetReport, split: Split) -> None:
    expected = {
        "dev": SplitCounts(answerable=15, unanswerable=2),
        "test": SplitCounts(answerable=5, unanswerable=2),
    }
    for name in ("dev", "test"):
        if split not in {name, "all"}:
            continue
        actual = getattr(report, name)
        minimum = expected[name]
        if actual.answerable < minimum.answerable or actual.unanswerable < minimum.unanswerable:
            raise ContractError(
                f"benchmark {name}: requires at least {minimum.answerable} answerable and "
                f"{minimum.unanswerable} unanswerable questions; found "
                f"{actual.answerable} and {actual.unanswerable}"
            )


def validate_dataset(
    chunk_values: Sequence[object],
    question_values: Sequence[object],
    *,
    split: Split = "all",
    require_minimums: bool = False,
) -> DatasetReport:
    """Validate records and evidence links without returning or logging benchmark gold."""
    if not chunk_values:
        raise ContractError("chunks: expected at least one record")
    if not question_values:
        raise ContractError("questions: expected at least one record")
    chunks = [Chunk.from_json(row, f"chunks[{index}]") for index, row in enumerate(chunk_values)]
    questions = [
        BenchmarkQuestion.from_json(row, f"questions[{index}]")
        for index, row in enumerate(question_values)
    ]
    _unique_ids(chunks, "chunks")
    _unique_ids(questions, "questions")

    selected = questions if split == "all" else [q for q in questions if q.split == split]
    if not selected:
        raise ContractError(f"questions: no records found for split {split!r}")
    for question in selected:
        if question.answerable:
            _gold_resolves(question, chunks)
        else:
            _absence_terms_do_not_resolve(question, chunks)

    report = DatasetReport(
        chunks=len(chunks),
        questions=len(selected),
        dev=_counts(selected, "dev"),
        test=_counts(selected, "test"),
    )
    if require_minimums:
        _require_minimums(report, split)
    return report
