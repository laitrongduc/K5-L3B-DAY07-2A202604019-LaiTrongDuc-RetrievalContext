"""Immutable benchmark manifest creation and verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .domain import ContractError
from .validation import load_json_array, validate_dataset


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractError(f"{path}: could not read file: {error}") from error


def freeze_benchmark(chunks_path: Path, questions_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate a complete benchmark and write a non-secret integrity manifest."""
    if output_path.exists():
        raise ContractError(f"refusing to overwrite existing freeze manifest: {output_path}")
    report = validate_dataset(
        load_json_array(chunks_path),
        load_json_array(questions_path),
        split="all",
        require_minimums=True,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "chunks": {
            "path": os.path.relpath(chunks_path.resolve(), output_path.parent.resolve()),
            "sha256": _sha256(chunks_path),
        },
        "questions": {
            "path": os.path.relpath(questions_path.resolve(), output_path.parent.resolve()),
            "sha256": _sha256(questions_path),
        },
        "counts": report.to_json(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def verify_benchmark_freeze(manifest_path: Path) -> dict[str, Any]:
    """Verify frozen file checksums and return only manifest metadata."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{manifest_path}: invalid freeze manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ContractError(f"{manifest_path}: unsupported freeze manifest schema")
    for key in ("chunks", "questions"):
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            raise ContractError(f"{manifest_path}: missing {key} entry")
        relative_path = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ContractError(f"{manifest_path}: invalid {key} path/checksum")
        actual = _sha256(manifest_path.parent / relative_path)
        if actual != expected:
            raise ContractError(
                f"{key} checksum mismatch: expected {expected}, found {actual}; "
                "do not tune against a modified frozen benchmark"
            )
    return manifest
