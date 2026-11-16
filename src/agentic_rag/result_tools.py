"""Validated utilities for composing immutable benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import ContractError


def _load_run(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load agent run {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("traces"), list):
        raise ContractError(f"agent run must contain a trace list: {path}")
    traces = value["traces"]
    ids = [trace.get("query_id") for trace in traces if isinstance(trace, dict)]
    if len(ids) != len(traces) or any(not isinstance(item, str) for item in ids):
        raise ContractError(f"agent run contains an invalid trace: {path}")
    if len(set(ids)) != len(ids):
        raise ContractError(f"agent run contains duplicate query IDs: {path}")
    return value


def merge_agent_runs(
    base_path: Path, repair_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    """Replace selected base traces with later validated repair traces."""
    if output_path.exists():
        raise ContractError(f"refusing to overwrite merged agent run: {output_path}")
    base = _load_run(base_path)
    traces = {trace["query_id"]: trace for trace in base["traces"]}
    order = list(traces)
    source_runs = [str(base_path)]
    elapsed = float(base.get("elapsed_seconds", 0.0))
    for repair_path in repair_paths:
        repair = _load_run(repair_path)
        if repair.get("split") != base.get("split") or repair.get("model") != base.get("model"):
            raise ContractError(f"repair run split/model mismatch: {repair_path}")
        for trace in repair["traces"]:
            query_id = trace["query_id"]
            if query_id not in traces:
                raise ContractError(f"repair query ID is absent from base run: {query_id}")
            traces[query_id] = trace
        source_runs.append(str(repair_path))
        elapsed += float(repair.get("elapsed_seconds", 0.0))
    report = {
        "split": base.get("split"),
        "model": base.get("model"),
        "cases": len(order),
        "elapsed_seconds": round(elapsed, 6),
        "source_runs": source_runs,
        "traces": [traces[query_id] for query_id in order],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def export_agent_traces(run_path: Path, output_path: Path) -> int:
    """Export the trace array required by the supplied metric aggregator."""
    if output_path.exists():
        raise ContractError(f"refusing to overwrite exported traces: {output_path}")
    run = _load_run(run_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run["traces"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(run["traces"])
