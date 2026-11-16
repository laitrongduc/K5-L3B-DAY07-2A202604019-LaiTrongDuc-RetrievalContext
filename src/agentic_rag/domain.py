"""Typed domain objects and fail-fast JSON contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]
Page = int | str


class ContractError(ValueError):
    """Raised when an input violates a documented data contract."""


def _object(value: object, location: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{location}: expected a JSON object with string keys")
    return value


def _text(row: JsonObject, key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{location}.{key}: expected a non-empty string")
    return value.strip()


def _content(row: JsonObject, key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{location}.{key}: expected non-empty text content")
    return value


def _page(row: JsonObject, location: str) -> Page:
    value = row.get("page")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ContractError(f"{location}.page: expected an integer or non-empty string")
    if isinstance(value, int) and value < 1:
        raise ContractError(f"{location}.page: expected an integer >= 1")
    if isinstance(value, str) and not value.strip():
        raise ContractError(f"{location}.page: expected a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    source_id: str
    source: str
    page: Page
    metadata: JsonObject

    @classmethod
    def from_json(cls, value: object, location: str) -> Chunk:
        row = _object(value, location)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise ContractError(f"{location}.metadata: expected a JSON object")
        return cls(
            id=_text(row, "id", location),
            text=_content(row, "text", location),
            source_id=_text(row, "source_id", location),
            source=_text(row, "source", location),
            page=_page(row, location),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    source_id: str
    page: Page
    text: str

    @classmethod
    def from_json(cls, value: object, location: str) -> Evidence:
        row = _object(value, location)
        return cls(
            source_id=_text(row, "source_id", location),
            page=_page(row, location),
            text=_content(row, "evidence", location),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    id: str
    split: str
    question: str
    answerable: bool
    expected_answer: str | None
    gold: tuple[Evidence, ...]
    absence_reason: str | None
    absence_terms: tuple[str, ...]
    filters: JsonObject | None

    @classmethod
    def from_json(cls, value: object, location: str) -> BenchmarkQuestion:
        row = _object(value, location)
        split = _text(row, "split", location)
        if split not in {"dev", "test"}:
            raise ContractError(f"{location}.split: expected 'dev' or 'test', got {split!r}")

        raw_gold = row.get("gold", [])
        if not isinstance(raw_gold, list):
            raise ContractError(f"{location}.gold: expected a JSON array")
        inferred_answerable = bool(raw_gold)
        answerable = row.get("answerable", inferred_answerable)
        if not isinstance(answerable, bool):
            raise ContractError(f"{location}.answerable: expected a boolean")

        gold = tuple(
            Evidence.from_json(item, f"{location}.gold[{index}]")
            for index, item in enumerate(raw_gold)
        )
        expected_answer = row.get("expected_answer")
        absence_reason = row.get("absence_reason")
        raw_absence_terms = row.get("absence_terms", [])
        if not isinstance(raw_absence_terms, list) or any(
            not isinstance(term, str) or not term.strip() for term in raw_absence_terms
        ):
            raise ContractError(f"{location}.absence_terms: expected non-empty strings")
        absence_terms = tuple(term.strip() for term in raw_absence_terms)
        if answerable:
            if not gold:
                raise ContractError(f"{location}.gold: answerable questions require evidence")
            if not isinstance(expected_answer, str) or not expected_answer.strip():
                raise ContractError(
                    f"{location}.expected_answer: answerable questions require an answer"
                )
            expected_answer = expected_answer.strip()
            absence_reason = None
            absence_terms = ()
        else:
            if gold:
                raise ContractError(
                    f"{location}.gold: unanswerable questions must not contain gold"
                )
            if not isinstance(absence_reason, str) or not absence_reason.strip():
                raise ContractError(
                    f"{location}.absence_reason: unanswerable questions require an audit reason"
                )
            if row.get("expected_behavior") != "search_and_abstain":
                raise ContractError(f"{location}.expected_behavior: expected 'search_and_abstain'")
            expected_answer = None
            absence_reason = absence_reason.strip()

        raw_filters = row.get("filter")
        if raw_filters is not None and (
            not isinstance(raw_filters, dict)
            or not all(isinstance(key, str) for key in raw_filters)
        ):
            raise ContractError(f"{location}.filter: expected a JSON object")

        return cls(
            id=_text(row, "id", location),
            split=split,
            question=_text(row, "question", location),
            answerable=answerable,
            expected_answer=expected_answer,
            gold=gold,
            absence_reason=absence_reason,
            absence_terms=absence_terms,
            filters=dict(raw_filters) if raw_filters is not None else None,
        )
