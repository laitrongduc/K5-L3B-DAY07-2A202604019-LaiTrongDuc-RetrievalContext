"""Run the real agent over a benchmark split without exposing gold to runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .agent_runtime import run_agent
from .controller import OpenRouterController, load_agent_config
from .domain import BenchmarkQuestion, ContractError
from .retrieval import DenseRetriever, HybridRRFRetriever, ReferenceAwareBM25Retriever, Retriever
from .validation import load_json_array


def run_agent_split(
    chunks_path: Path,
    questions_path: Path,
    agent_config_path: Path,
    embedding_config_path: Path,
    output_path: Path,
    *,
    split: str,
    question_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite agent traces: {output_path}")
    config = load_agent_config(agent_config_path)
    questions = [
        BenchmarkQuestion.from_json(value, f"questions[{index}]")
        for index, value in enumerate(load_json_array(questions_path))
    ]
    selected = [question for question in questions if question.split == split]
    if question_ids:
        requested = set(question_ids)
        selected = [question for question in selected if question.id in requested]
        missing = sorted(requested - {question.id for question in selected})
        if missing:
            raise ContractError(
                f"question IDs are absent from split {split!r}: {', '.join(missing)}"
            )
    lexical = ReferenceAwareBM25Retriever(chunks_path, candidate_k=config.candidate_k)
    dense = DenseRetriever(embedding_config_path)
    hybrid = HybridRRFRetriever(lexical, dense)
    tools: dict[str, Retriever] = {
        "search_lexical": lexical,
        "search_dense": dense,
        "search_hybrid": hybrid,
    }
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    for question in selected:
        controller = OpenRouterController(config)
        case_started = time.perf_counter()
        try:
            result = run_agent(
                question.question,
                controller,
                tools,
                max_searches=config.max_search_calls,
                top_k=config.context_k,
                context_max_characters=config.context_max_characters,
            )
            result.update(
                {
                    "query_id": question.id,
                    "model_calls": controller.calls,
                    "elapsed_seconds": round(time.perf_counter() - case_started, 6),
                }
            )
        except Exception as error:  # provider errors belong in benchmark output
            result = {
                "query_id": question.id,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "events": [],
                "citations": [],
                "model_calls": controller.calls,
                "elapsed_seconds": round(time.perf_counter() - case_started, 6),
            }
        traces.append(result)
        checkpoint = {
            "split": split,
            "model": config.model,
            "cases": len(selected),
            "completed_cases": len(traces),
            "traces": traces,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    report = {
        "split": split,
        "model": config.model,
        "cases": len(selected),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "traces": traces,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
