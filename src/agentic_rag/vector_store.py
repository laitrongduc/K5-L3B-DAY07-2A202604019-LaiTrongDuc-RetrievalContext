"""Transactional pgvector indexing with deterministic runtime verification."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .domain import ContractError
from .validation import load_json_array


@dataclass(frozen=True, slots=True)
class VectorRow:
    chunk_id: str
    source_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...]


def _vector_literal(vector: tuple[float, ...]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in vector) + "]"


def load_vector_rows(chunks_path: Path, embeddings_path: Path, dimensions: int) -> list[VectorRow]:
    chunks = load_json_array(chunks_path)
    embeddings = load_json_array(embeddings_path)
    by_id: dict[str, tuple[float, ...]] = {}
    for index, item in enumerate(embeddings):
        if not isinstance(item, dict):
            raise ContractError(f"embedding[{index}]: expected object")
        chunk_id, raw = item.get("chunk_id"), item.get("embedding")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in by_id:
            raise ContractError(f"embedding[{index}]: invalid or duplicate chunk_id")
        if not isinstance(raw, list) or len(raw) != dimensions:
            raise ContractError(f"embedding[{index}]: expected vector({dimensions})")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in raw):
            raise ContractError(f"embedding[{index}]: vector contains invalid value")
        vector = tuple(float(value) for value in raw)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ContractError(f"embedding[{index}]: expected normalized vector, norm={norm}")
        by_id[chunk_id] = vector

    rows: list[VectorRow] = []
    chunk_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ContractError(f"chunk[{index}]: expected object")
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in chunk_ids:
            raise ContractError(f"chunk[{index}]: invalid or duplicate id")
        chunk_ids.add(chunk_id)
        source_id, source, page = chunk.get("source_id"), chunk.get("source"), chunk.get("page")
        text, metadata = chunk.get("text"), chunk.get("metadata")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(source, str)
            or not source
            or type(page) is not int
            or page < 1
            or not isinstance(text, str)
            or not text
            or not isinstance(metadata, dict)
        ):
            raise ContractError(f"chunk[{index}]: invalid pgvector metadata")
        stored_vector = by_id.get(chunk_id)
        if stored_vector is None:
            raise ContractError(f"chunk[{index}]: missing embedding for {chunk_id}")
        rows.append(VectorRow(chunk_id, source_id, source, page, text, metadata, stored_vector))
    extras = sorted(set(by_id) - chunk_ids)
    if extras:
        raise ContractError(f"embeddings contain unknown chunk IDs: {extras[:3]}")
    return rows


def index_pgvector(
    chunks_path: Path,
    embeddings_path: Path,
    *,
    dimensions: int,
    model: str,
    database_url: str | None = None,
) -> dict[str, Any]:
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL", "")
    if not url.strip():
        raise ContractError("DATABASE_URL is required for pgvector indexing")
    if dimensions < 1 or not model.strip():
        raise ContractError("pgvector dimensions/model must be configured")
    rows = load_vector_rows(chunks_path, embeddings_path, dimensions)
    try:
        with psycopg.connect(url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS corpus_chunks (
                        chunk_id text PRIMARY KEY,
                        source_id text NOT NULL,
                        source text NOT NULL,
                        page integer NOT NULL CHECK (page > 0),
                        content text NOT NULL CHECK (length(content) > 0),
                        metadata jsonb NOT NULL,
                        embedding vector({dimensions}) NOT NULL,
                        embedding_model text NOT NULL
                    )
                    """
                )
                cursor.execute("SELECT count(*) FROM corpus_chunks")
                existing = cursor.fetchone()
                if existing is None or existing[0] != 0:
                    count = "unknown" if existing is None else existing[0]
                    raise ContractError(
                        f"refusing to index into non-empty corpus_chunks table ({count} rows)"
                    )
                cursor.executemany(
                    """
                    INSERT INTO corpus_chunks (
                        chunk_id, source_id, source, page, content, metadata,
                        embedding, embedding_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    """,
                    [
                        (
                            row.chunk_id,
                            row.source_id,
                            row.source,
                            row.page,
                            row.text,
                            Jsonb(row.metadata),
                            _vector_literal(row.embedding),
                            model,
                        )
                        for row in rows
                    ],
                )
        with psycopg.connect(url) as verification:
            with verification.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*), count(DISTINCT chunk_id),
                           min(vector_dims(embedding)), max(vector_dims(embedding))
                    FROM corpus_chunks
                    """
                )
                counts = cursor.fetchone()
                if counts != (len(rows), len(rows), dimensions, dimensions):
                    raise ContractError(f"pgvector verification mismatch: {counts}")
                probe = rows[0]
                cursor.execute(
                    """
                    SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                    FROM corpus_chunks ORDER BY embedding <=> %s::vector, chunk_id LIMIT 1
                    """,
                    (_vector_literal(probe.embedding), _vector_literal(probe.embedding)),
                )
                nearest = cursor.fetchone()
                if (
                    nearest is None
                    or nearest[0] != probe.chunk_id
                    or not math.isclose(float(nearest[1]), 1.0, abs_tol=1e-6)
                ):
                    raise ContractError(f"pgvector reopen search verification failed: {nearest}")
    except psycopg.Error as error:
        raise ContractError(f"pgvector database error: {error}") from error
    return {
        "table": "corpus_chunks",
        "rows": len(rows),
        "unique_chunk_ids": len(rows),
        "dimensions": dimensions,
        "embedding_model": model,
        "reopen_probe_chunk_id": rows[0].chunk_id,
        "reopen_probe_score": 1.0,
    }
