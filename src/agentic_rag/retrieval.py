"""Lexical, dense, and hybrid retrieval over the same traceable corpus."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psycopg

from .domain import ContractError
from .embeddings import OpenRouterEmbedder, load_embedding_config
from .validation import load_json_array

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    rank: int
    method: str


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]: ...


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def _validate_query(query: str, top_k: int) -> str:
    normalized = query.strip()
    if not normalized:
        raise ContractError("retrieval query must not be empty")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ContractError("retrieval top_k must be an integer in [1, 100]")
    return normalized


class BM25Retriever:
    def __init__(self, chunks_path: Path, *, k1: float = 1.5, b: float = 0.75) -> None:
        chunks = load_json_array(chunks_path)
        self._rows: list[dict[str, Any]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._lengths: list[int] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ContractError(f"chunk[{index}]: expected object")
            chunk_id, text = chunk.get("id"), chunk.get("text")
            source, page = chunk.get("source"), chunk.get("page")
            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or not isinstance(text, str)
                or not text
                or not isinstance(source, str)
                or not source
                or type(page) is not int
                or page < 1
            ):
                raise ContractError(f"chunk[{index}]: invalid retrieval data")
            tokens = _tokens(text)
            frequencies = Counter(tokens)
            self._rows.append(chunk)
            self._term_frequencies.append(frequencies)
            self._document_frequency.update(frequencies.keys())
            self._lengths.append(len(tokens))
        if not self._rows or not sum(self._lengths):
            raise ContractError("BM25 corpus must contain tokenizable chunks")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ContractError("BM25 requires k1 > 0 and b in [0, 1]")
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._k1 = k1
        self._b = b

    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        normalized = _validate_query(query, top_k)
        terms = set(_tokens(normalized))
        if not terms:
            raise ContractError("retrieval query has no tokenizable terms")
        scored: list[tuple[float, str, dict[str, Any]]] = []
        documents = len(self._rows)
        for row, frequencies, length in zip(
            self._rows, self._term_frequencies, self._lengths, strict=True
        ):
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse = math.log(
                    1 + (documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / self._average_length
                )
                score += inverse * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((score, str(row["id"]), row))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return [
            RetrievalResult(
                chunk_id=str(row["id"]),
                text=str(row["text"]),
                source=str(row["source"]),
                page=int(row["page"]),
                score=score,
                rank=rank,
                method="bm25",
            )
            for rank, (score, _chunk_id, row) in enumerate(scored[:top_k], start=1)
        ]


class ReferenceAwareBM25Retriever:
    """BM25 with deterministic TOC suppression for explicit figure references."""

    def __init__(self, chunks_path: Path, *, candidate_k: int = 20) -> None:
        self._base = BM25Retriever(chunks_path)
        self._candidate_k = candidate_k
        self._chapter: dict[str, int] = {}
        for value in load_json_array(chunks_path):
            if isinstance(value, dict) and isinstance(value.get("id"), str):
                metadata = value.get("metadata", {})
                chapter = metadata.get("chapter", 0) if isinstance(metadata, dict) else 0
                self._chapter[value["id"]] = chapter if type(chapter) is int else 0

    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        candidates = self._base.search(query, top_k=max(top_k, self._candidate_k))
        match = re.search(r"\bhình\s+(\d+)\b", query, flags=re.IGNORECASE)
        if match is not None:
            caption = re.compile(rf"Hình\s+vẽ\s+{re.escape(match.group(1))}\s*:", re.IGNORECASE)
            preferred = [
                result
                for result in candidates
                if self._chapter.get(result.chunk_id, 0) != 0 and caption.search(result.text)
            ]
            preferred_ids = {result.chunk_id for result in preferred}
            candidates = preferred + [
                result for result in candidates if result.chunk_id not in preferred_ids
            ]
        return [
            RetrievalResult(
                result.chunk_id,
                result.text,
                result.source,
                result.page,
                result.score,
                rank,
                "bm25_reference_aware",
            )
            for rank, result in enumerate(candidates[:top_k], start=1)
        ]


class DenseRetriever:
    def __init__(
        self,
        embedding_config_path: Path,
        *,
        database_url: str | None = None,
    ) -> None:
        self._config = load_embedding_config(embedding_config_path)
        self._embedder = OpenRouterEmbedder(self._config)
        self._database_url = database_url or os.environ.get("DATABASE_URL", "")
        if not self._database_url.strip():
            raise ContractError("DATABASE_URL is required for dense retrieval")

    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        normalized = _validate_query(query, top_k)
        batch = self._embedder.embed([normalized], input_type="search_query")
        vector = batch.vectors[0]
        literal = "[" + ",".join(format(value, ".17g") for value in vector) + "]"
        try:
            with psycopg.connect(self._database_url) as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, content, source, page,
                           1 - (embedding <=> %s::vector) AS score
                    FROM corpus_chunks
                    WHERE embedding_model = %s
                    ORDER BY embedding <=> %s::vector, chunk_id
                    LIMIT %s
                    """,
                    (literal, self._config.model, literal, top_k),
                )
                rows = cursor.fetchall()
        except psycopg.Error as error:
            raise ContractError(f"dense retrieval database error: {error}") from error
        return [
            RetrievalResult(
                chunk_id=str(row[0]),
                text=str(row[1]),
                source=str(row[2]),
                page=int(row[3]),
                score=float(row[4]),
                rank=rank,
                method="dense",
            )
            for rank, row in enumerate(rows, start=1)
        ]


class HybridRRFRetriever:
    def __init__(self, lexical: Retriever, dense: Retriever, *, rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ContractError("RRF k must be positive")
        self._lexical = lexical
        self._dense = dense
        self._rrf_k = rrf_k

    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        normalized = _validate_query(query, top_k)
        candidate_k = min(max(top_k * 4, 20), 100)
        candidates = (
            self._lexical.search(normalized, top_k=candidate_k),
            self._dense.search(normalized, top_k=candidate_k),
        )
        scores: dict[str, float] = {}
        by_id: dict[str, RetrievalResult] = {}
        for ranking in candidates:
            for result in ranking:
                by_id.setdefault(result.chunk_id, result)
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (
                    self._rrf_k + result.rank
                )
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                text=by_id[chunk_id].text,
                source=by_id[chunk_id].source,
                page=by_id[chunk_id].page,
                score=scores[chunk_id],
                rank=rank,
                method="hybrid_rrf",
            )
            for rank, chunk_id in enumerate(ordered, start=1)
        ]
