"""Create reproducible corpus embedding artifacts from validated chunks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .domain import ContractError
from .embeddings import OpenRouterEmbedder, load_embedding_config
from .validation import load_json_array

_BGE_M3_PRICE_PER_MILLION_TOKENS_USD = 0.01
_PRICE_CHECKED_DATE = "2026-11-16"


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractError(f"{path}: could not calculate checksum: {error}") from error


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def embed_corpus(chunks_path: Path, config_path: Path, output_directory: Path) -> dict[str, Any]:
    embeddings_path = output_directory / "embeddings.json"
    manifest_path = output_directory / "embedding_manifest.json"
    existing = [path for path in (embeddings_path, manifest_path) if path.exists()]
    if existing:
        raise ContractError(
            "refusing to overwrite embedding output: " + ", ".join(str(path) for path in existing)
        )
    chunks = load_json_array(chunks_path)
    if not chunks:
        raise ContractError("embedding corpus must contain at least one chunk")
    chunk_ids: list[str] = []
    texts: list[str] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ContractError(f"chunk[{index}]: expected object")
        chunk_id, text = chunk.get("id"), chunk.get("text")
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ContractError(f"chunk[{index}]: embedding requires non-empty id and text")
        chunk_ids.append(chunk_id)
        texts.append(text)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ContractError("embedding corpus contains duplicate chunk IDs")

    config = load_embedding_config(config_path)
    started = time.perf_counter()
    result = OpenRouterEmbedder(config).embed(texts, input_type="search_document")
    elapsed_seconds = time.perf_counter() - started
    if len(result.vectors) != len(chunks):
        raise ContractError("embedding adapter returned a different number of vectors than chunks")
    rows = [
        {"chunk_id": chunk_id, "embedding": list(vector)}
        for chunk_id, vector in zip(chunk_ids, result.vectors, strict=True)
    ]
    estimated_cost = result.prompt_tokens / 1_000_000 * _BGE_M3_PRICE_PER_MILLION_TOKENS_USD
    manifest: dict[str, Any] = {
        "chunks_path": str(chunks_path),
        "chunks_sha256": _sha256(chunks_path),
        "embedding_config_sha256": _sha256(config_path),
        "requested_model": config.model,
        "response_models": list(result.response_models),
        "dimensions": config.dimensions,
        "normalized": config.normalize,
        "vectors": len(rows),
        "prompt_tokens": result.prompt_tokens,
        "total_tokens": result.total_tokens,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "estimated_cost_usd": round(estimated_cost, 8),
        "price_assumption_usd_per_million_tokens": _BGE_M3_PRICE_PER_MILLION_TOKENS_USD,
        "price_checked_date": _PRICE_CHECKED_DATE,
    }
    _atomic_json(embeddings_path, rows)
    manifest["embeddings_sha256"] = _sha256(embeddings_path)
    _atomic_json(manifest_path, manifest)
    return manifest
