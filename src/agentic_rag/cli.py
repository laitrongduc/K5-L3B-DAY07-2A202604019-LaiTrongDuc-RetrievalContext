"""Reusable command-line interface for deterministic project workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .agent_runner import run_agent_split
from .domain import ContractError
from .embedding_pipeline import embed_corpus
from .evaluation import evaluate_dev_retrieval, evaluate_retrieval
from .freezing import freeze_benchmark, verify_benchmark_freeze
from .ingestion import prepare_corpus
from .quality import run_quality_checks
from .result_tools import export_agent_traces, merge_agent_runs
from .validation import load_json_array, validate_dataset
from .vector_store import index_pgvector


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate-data", help="validate chunk/question contracts without exposing gold"
    )
    validate.add_argument("--chunks", type=Path, required=True)
    validate.add_argument("--questions", type=Path, required=True)
    validate.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    validate.add_argument(
        "--require-minimums",
        action="store_true",
        help="enforce the Day 7 benchmark minimum counts for the selected split",
    )
    freeze = commands.add_parser("freeze-benchmark", help="freeze validated benchmark checksums")
    freeze.add_argument("--chunks", type=Path, required=True)
    freeze.add_argument("--questions", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify_freeze = commands.add_parser(
        "verify-freeze", help="verify that frozen benchmark files are unchanged"
    )
    verify_freeze.add_argument("--manifest", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-corpus", help="extract and chunk an authorized PDF corpus"
    )
    prepare.add_argument("--pdf", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    embed = commands.add_parser(
        "embed-corpus", help="create real, validated OpenRouter embeddings for corpus chunks"
    )
    embed.add_argument("--chunks", type=Path, required=True)
    embed.add_argument("--config", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    index = commands.add_parser(
        "index-pgvector", help="transactionally load and verify corpus embeddings in pgvector"
    )
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--embeddings", type=Path, required=True)
    index.add_argument("--dimensions", type=int, required=True)
    index.add_argument("--model", required=True)
    evaluate = commands.add_parser(
        "evaluate-dev-retrieval", help="evaluate frozen dev questions without reading test gold"
    )
    evaluate.add_argument("--chunks", type=Path, required=True)
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--embedding-config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate_split = commands.add_parser(
        "evaluate-retrieval", help="evaluate a frozen retrieval configuration on one split"
    )
    evaluate_split.add_argument("--chunks", type=Path, required=True)
    evaluate_split.add_argument("--questions", type=Path, required=True)
    evaluate_split.add_argument("--embedding-config", type=Path, required=True)
    evaluate_split.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate_split.add_argument("--output", type=Path, required=True)
    run_split = commands.add_parser("run-agent-split", help="run the real bounded agent")
    run_split.add_argument("--chunks", type=Path, required=True)
    run_split.add_argument("--questions", type=Path, required=True)
    run_split.add_argument("--agent-config", type=Path, required=True)
    run_split.add_argument("--embedding-config", type=Path, required=True)
    run_split.add_argument("--split", choices=("dev", "test"), required=True)
    run_split.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="run only this ID; repeat for multiple cases (useful for deterministic repair checks)",
    )
    run_split.add_argument("--output", type=Path, required=True)
    merge_runs = commands.add_parser(
        "merge-agent-runs", help="replace selected base traces with validated repair runs"
    )
    merge_runs.add_argument("--base", type=Path, required=True)
    merge_runs.add_argument("--repair", type=Path, action="append", required=True)
    merge_runs.add_argument("--output", type=Path, required=True)
    export_traces = commands.add_parser(
        "export-agent-traces", help="export a run's trace array for metric aggregation"
    )
    export_traces.add_argument("--run", type=Path, required=True)
    export_traces.add_argument("--output", type=Path, required=True)
    commands.add_parser("verify", help="run format, lint, type, compile, and test gates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            validation_report = validate_dataset(
                load_json_array(args.chunks),
                load_json_array(args.questions),
                split=args.split,
                require_minimums=args.require_minimums,
            )
            print(
                json.dumps(
                    validation_report.to_json(), ensure_ascii=False, indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "verify":
            root = Path(__file__).resolve().parents[2]
            return run_quality_checks(root)
        if args.command == "freeze-benchmark":
            manifest = freeze_benchmark(args.chunks, args.questions, args.output)
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "verify-freeze":
            manifest = verify_benchmark_freeze(args.manifest)
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "prepare-corpus":
            stats = prepare_corpus(args.pdf, args.config, args.output)
            print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "embed-corpus":
            manifest = embed_corpus(args.chunks, args.config, args.output)
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "index-pgvector":
            report = index_pgvector(
                args.chunks,
                args.embeddings,
                dimensions=args.dimensions,
                model=args.model,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate-dev-retrieval":
            report = evaluate_dev_retrieval(
                args.chunks, args.questions, args.embedding_config, args.output
            )
            summary = {
                method["method"]: {
                    key: method[key]
                    for key in (
                        "hit_at_1",
                        "evidence_hit_at_3",
                        "context_evidence_rate",
                        "elapsed_seconds",
                    )
                }
                for method in report["methods"]
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate-retrieval":
            report = evaluate_retrieval(
                args.chunks,
                args.questions,
                args.embedding_config,
                args.output,
                split=args.split,
            )
            summary = {
                method["method"]: {
                    key: method[key]
                    for key in (
                        "hit_at_1",
                        "evidence_hit_at_3",
                        "context_evidence_rate",
                        "elapsed_seconds",
                    )
                }
                for method in report["methods"]
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "run-agent-split":
            report = run_agent_split(
                args.chunks,
                args.questions,
                args.agent_config,
                args.embedding_config,
                args.output,
                split=args.split,
                question_ids=tuple(args.question_id),
            )
            print(
                json.dumps(
                    {
                        "split": report["split"],
                        "model": report["model"],
                        "cases": report["cases"],
                        "elapsed_seconds": report["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "merge-agent-runs":
            report = merge_agent_runs(args.base, args.repair, args.output)
            print(
                json.dumps(
                    {key: report[key] for key in ("split", "model", "cases", "source_runs")},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "export-agent-traces":
            count = export_agent_traces(args.run, args.output)
            print(json.dumps({"traces": count, "output": str(args.output)}, indent=2))
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except ContractError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
