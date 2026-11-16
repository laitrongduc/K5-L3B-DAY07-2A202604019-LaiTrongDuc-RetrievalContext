"""Dev-only retrieval evaluation with gold isolated from runtime retrievers."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import psycopg

from .domain import BenchmarkQuestion, Chunk, ContractError
from .embeddings import OpenRouterEmbedder, load_embedding_config
from .retrieval import BM25Retriever, RetrievalResult
from .validation import load_json_array


def _gold_chunk_ids(question: BenchmarkQuestion, chunks: list[Chunk]) -> set[str]:
    targets: set[str] = set()
    for evidence in question.gold:
        page_chunks = [
            chunk
            for chunk in chunks
            if chunk.source_id == evidence.source_id and chunk.page == evidence.page
        ]
        direct = {chunk.id for chunk in page_chunks if evidence.text in chunk.text}
        if direct:
            targets.update(direct)
            continue
        spans: list[tuple[int, int, Chunk]] = []
        for chunk in page_chunks:
            start, end = chunk.metadata.get("start_char"), chunk.metadata.get("end_char")
            if type(start) is int and type(end) is int and end > start:
                spans.append((start, end, chunk))
        if not spans:
            raise ContractError(f"question {question.id}: evidence has no traceable chunk spans")
        page = ["\0"] * max(end for _, end, _ in spans)
        for start, _end, chunk in spans:
            for offset, character in enumerate(chunk.text, start=start):
                if page[offset] not in {"\0", character}:
                    raise ContractError(f"question {question.id}: conflicting page reconstruction")
                page[offset] = character
        reconstructed = "".join(page)
        evidence_start = reconstructed.find(evidence.text)
        if evidence_start < 0:
            raise ContractError(
                f"question {question.id}: evidence did not resolve during evaluation"
            )
        evidence_end = evidence_start + len(evidence.text)
        targets.update(
            chunk.id for start, end, chunk in spans if start < evidence_end and end > evidence_start
        )
    return targets


def _dense_rankings(
    questions: list[BenchmarkQuestion],
    vectors: tuple[tuple[float, ...], ...],
    *,
    database_url: str,
    model: str,
    top_k: int,
) -> dict[str, list[RetrievalResult]]:
    rankings: dict[str, list[RetrievalResult]] = {}
    try:
        with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
            for question, vector in zip(questions, vectors, strict=True):
                literal = "[" + ",".join(format(value, ".17g") for value in vector) + "]"
                cursor.execute(
                    """
                    SELECT chunk_id, content, source, page,
                           1 - (embedding <=> %s::vector) AS score
                    FROM corpus_chunks WHERE embedding_model = %s
                    ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s
                    """,
                    (literal, model, literal, top_k),
                )
                rankings[question.id] = [
                    RetrievalResult(
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                        int(row[3]),
                        float(row[4]),
                        rank,
                        "dense",
                    )
                    for rank, row in enumerate(cursor.fetchall(), start=1)
                ]
    except psycopg.Error as error:
        raise ContractError(f"retrieval evaluation database error: {error}") from error
    return rankings


def _rrf(
    lexical: list[RetrievalResult],
    dense: list[RetrievalResult],
    *,
    top_k: int,
    rrf_k: int,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    rows: dict[str, RetrievalResult] = {}
    for ranking, weight in ((lexical, lexical_weight), (dense, dense_weight)):
        for result in ranking:
            rows.setdefault(result.chunk_id, result)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (
                rrf_k + result.rank
            )
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        RetrievalResult(
            chunk_id,
            rows[chunk_id].text,
            rows[chunk_id].source,
            rows[chunk_id].page,
            scores[chunk_id],
            rank,
            "hybrid_rrf",
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]


def _reference_aware(
    question: str,
    ranking: list[RetrievalResult],
    chapter_by_chunk: dict[str, int],
) -> list[RetrievalResult]:
    """Prefer body captions over table-of-contents entries for explicit figure references."""
    match = re.search(r"\bhình\s+(\d+)\b", question, flags=re.IGNORECASE)
    if match is None:
        return ranking
    caption = re.compile(rf"Hình\s+vẽ\s+{re.escape(match.group(1))}\s*:", re.IGNORECASE)
    body_matches = [
        result
        for result in ranking
        if chapter_by_chunk.get(result.chunk_id, 0) != 0 and caption.search(result.text)
    ]
    if not body_matches:
        return ranking
    matched_ids = {result.chunk_id for result in body_matches}
    ordered = body_matches + [result for result in ranking if result.chunk_id not in matched_ids]
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
        for rank, result in enumerate(ordered, start=1)
    ]


def _summarize(
    method: str,
    questions: list[BenchmarkQuestion],
    rankings: dict[str, list[RetrievalResult]],
    gold: dict[str, set[str]],
    *,
    elapsed_seconds: float,
    context_k: int = 3,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    hit_1 = hit_3 = context_hits = 0
    for question in questions:
        results = rankings[question.id]
        targets = gold[question.id]
        first = bool(results and results[0].chunk_id in targets)
        first_three = any(result.chunk_id in targets for result in results[:3])
        context_has_evidence = any(result.chunk_id in targets for result in results[:context_k])
        hit_1 += first
        hit_3 += first_three
        context_hits += context_has_evidence
        traces.append(
            {
                "question_id": question.id,
                "hit_at_1": first,
                "evidence_hit_at_3": first_three,
                "context_has_evidence": context_has_evidence,
                "results": [
                    {
                        "chunk_id": result.chunk_id,
                        "page": result.page,
                        "rank": result.rank,
                        "score": round(result.score, 8),
                    }
                    for result in results[:3]
                ],
            }
        )
    total = len(questions)
    return {
        "method": method,
        "answerable_questions": total,
        "hit_at_1": {"n": hit_1, "N": total, "rate": hit_1 / total},
        "evidence_hit_at_3": {"n": hit_3, "N": total, "rate": hit_3 / total},
        "context_evidence_rate": {
            "n": context_hits,
            "N": total,
            "rate": context_hits / total,
            "context_k": context_k,
        },
        "elapsed_seconds": round(elapsed_seconds, 6),
        "traces": traces,
    }


def evaluate_retrieval(
    chunks_path: Path,
    questions_path: Path,
    embedding_config_path: Path,
    output_path: Path,
    *,
    split: str,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    if output_path.exists():
        raise ContractError(f"refusing to overwrite retrieval evaluation: {output_path}")
    chunk_values = load_json_array(chunks_path)
    question_values = load_json_array(questions_path)
    chunks = [
        Chunk.from_json(value, f"chunks[{index}]") for index, value in enumerate(chunk_values)
    ]
    all_questions = [
        BenchmarkQuestion.from_json(value, f"questions[{index}]")
        for index, value in enumerate(question_values)
    ]
    if split not in {"dev", "test"}:
        raise ContractError("retrieval evaluation split must be dev or test")
    questions = [
        question for question in all_questions if question.split == split and question.answerable
    ]
    minimum = 15 if split == "dev" else 5
    if len(questions) < minimum:
        raise ContractError(
            f"{split} retrieval evaluation requires at least {minimum} answerable questions"
        )
    gold = {question.id: _gold_chunk_ids(question, chunks) for question in questions}
    chapter_by_chunk = {
        chunk.id: int(chunk.metadata.get("chapter", 0))
        for chunk in chunks
        if type(chunk.metadata.get("chapter", 0)) is int
    }

    lexical = BM25Retriever(chunks_path)
    lexical_started = time.perf_counter()
    lexical_rankings = {
        question.id: lexical.search(question.question, top_k=candidate_k) for question in questions
    }
    lexical_elapsed = time.perf_counter() - lexical_started
    reference_started = time.perf_counter()
    reference_rankings = {
        question.id: _reference_aware(
            question.question, lexical_rankings[question.id], chapter_by_chunk
        )
        for question in questions
    }
    reference_elapsed = time.perf_counter() - reference_started

    config = load_embedding_config(embedding_config_path)
    dense_started = time.perf_counter()
    embedding_batch = OpenRouterEmbedder(config).embed(
        [question.question for question in questions], input_type="search_query"
    )
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        raise ContractError("DATABASE_URL is required for retrieval evaluation")
    dense_rankings = _dense_rankings(
        questions,
        embedding_batch.vectors,
        database_url=database_url,
        model=config.model,
        top_k=candidate_k,
    )
    dense_elapsed = time.perf_counter() - dense_started

    hybrid_started = time.perf_counter()
    hybrid_rankings = {
        question.id: _rrf(
            lexical_rankings[question.id],
            dense_rankings[question.id],
            top_k=candidate_k,
            rrf_k=rrf_k,
        )
        for question in questions
    }
    hybrid_elapsed = time.perf_counter() - hybrid_started
    weighted_started = time.perf_counter()
    weighted_rankings = {
        question.id: _rrf(
            reference_rankings[question.id],
            dense_rankings[question.id],
            top_k=candidate_k,
            rrf_k=rrf_k,
            lexical_weight=2.0,
            dense_weight=1.0,
        )
        for question in questions
    }
    weighted_elapsed = time.perf_counter() - weighted_started
    estimated_cost = embedding_batch.prompt_tokens / 1_000_000 * 0.01
    report = {
        "split": split,
        "candidate_k": candidate_k,
        "context_k": 3,
        "rrf_k": rrf_k,
        "query_embedding": {
            "requested_model": config.model,
            "response_models": list(embedding_batch.response_models),
            "prompt_tokens": embedding_batch.prompt_tokens,
            "total_tokens": embedding_batch.total_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
        },
        "methods": [
            _summarize("bm25", questions, lexical_rankings, gold, elapsed_seconds=lexical_elapsed),
            _summarize(
                "bm25_reference_aware",
                questions,
                reference_rankings,
                gold,
                elapsed_seconds=reference_elapsed,
            ),
            _summarize("dense", questions, dense_rankings, gold, elapsed_seconds=dense_elapsed),
            _summarize(
                "hybrid_rrf", questions, hybrid_rankings, gold, elapsed_seconds=hybrid_elapsed
            ),
            _summarize(
                "hybrid_rrf_lexical_2x",
                questions,
                weighted_rankings,
                gold,
                elapsed_seconds=weighted_elapsed,
            ),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def evaluate_dev_retrieval(
    chunks_path: Path,
    questions_path: Path,
    embedding_config_path: Path,
    output_path: Path,
    *,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Backward-compatible dev-only entry point."""
    return evaluate_retrieval(
        chunks_path,
        questions_path,
        embedding_config_path,
        output_path,
        split="dev",
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
