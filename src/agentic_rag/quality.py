"""Deterministic quality-gate orchestration for local development and CI."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def quality_commands() -> tuple[tuple[str, ...], ...]:
    """Return the ordered, fail-fast verification pipeline."""
    python = sys.executable
    code_paths = (
        "src/agentic_rag",
        "tests/solution/test_agent_runtime.py",
        "tests/solution/test_controller.py",
        "tests/solution/test_data_contract.py",
        "tests/solution/test_embeddings.py",
        "tests/solution/test_evaluation.py",
        "tests/solution/test_embedding_pipeline.py",
        "tests/solution/test_freezing.py",
        "tests/solution/test_ingestion.py",
        "tests/solution/test_project_cli.py",
        "tests/solution/test_quality.py",
        "tests/solution/test_result_tools.py",
        "tests/solution/test_retrieval.py",
        "tests/solution/test_vector_store.py",
    )
    return (
        (python, "-m", "ruff", "format", "--check", *code_paths),
        (python, "-m", "ruff", "check", *code_paths),
        (python, "-m", "mypy"),
        (python, "-m", "compileall", "-q", "src/agentic_rag", "starter"),
        (python, "-m", "unittest", "discover", "-s", "tests", "-v"),
    )


def run_quality_checks(root: Path, commands: Sequence[Sequence[str]] | None = None) -> int:
    """Run quality gates in order and return the first failing exit code."""
    selected = commands if commands is not None else quality_commands()
    for command in selected:
        printable = " ".join(command)
        print(f"[verify] {printable}", flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            print(f"[verify] failed ({result.returncode}): {printable}", file=sys.stderr)
            return result.returncode
    print("[verify] all quality gates passed")
    return 0
