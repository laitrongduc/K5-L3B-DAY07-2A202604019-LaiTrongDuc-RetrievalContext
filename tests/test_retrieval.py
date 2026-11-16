from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag.domain import ContractError
from agentic_rag.retrieval import (
    BM25Retriever,
    HybridRRFRetriever,
    ReferenceAwareBM25Retriever,
    RetrievalResult,
)


class StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        return self.results[:top_k]


class RetrievalTests(unittest.TestCase):
    def test_bm25_handles_vietnamese_and_stable_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.json"
            chunks.write_text(
                json.dumps(
                    [
                        {"id": "b", "text": "xác thực điện tử", "source": "s", "page": 2},
                        {"id": "a", "text": "định danh điện tử", "source": "s", "page": 1},
                        {"id": "c", "text": "nội dung khác", "source": "s", "page": 3},
                    ]
                ),
                encoding="utf-8",
            )
            results = BM25Retriever(chunks).search("điện tử", top_k=3)
        self.assertEqual(["a", "b"], [result.chunk_id for result in results])
        self.assertEqual([1, 2], [result.rank for result in results])

    def test_hybrid_rrf_combines_distinct_rankings(self) -> None:
        def result(chunk_id: str, rank: int, method: str) -> RetrievalResult:
            return RetrievalResult(chunk_id, chunk_id, "s", rank, 1.0, rank, method)

        lexical = StubRetriever([result("a", 1, "bm25"), result("b", 2, "bm25")])
        dense = StubRetriever([result("b", 1, "dense"), result("c", 2, "dense")])
        results = HybridRRFRetriever(lexical, dense).search("query", top_k=3)
        self.assertEqual("b", results[0].chunk_id)
        self.assertEqual("hybrid_rrf", results[0].method)

    def test_empty_query_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.json"
            chunks.write_text(
                json.dumps([{"id": "a", "text": "text", "source": "s", "page": 1}]),
                encoding="utf-8",
            )
            retriever = BM25Retriever(chunks)
            with self.assertRaisesRegex(ContractError, "must not be empty"):
                retriever.search(" ", top_k=3)

    def test_reference_aware_prefers_body_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks = Path(directory) / "chunks.json"
            chunks.write_text(
                json.dumps(
                    [
                        {
                            "id": "toc",
                            "text": "Hình vẽ 3: mục lục",
                            "source": "s",
                            "page": 2,
                            "metadata": {"chapter": 0},
                        },
                        {
                            "id": "body",
                            "text": "Hình vẽ 3: nội dung",
                            "source": "s",
                            "page": 20,
                            "metadata": {"chapter": 3},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows = ReferenceAwareBM25Retriever(chunks).search("Theo Hình 3?", top_k=2)
            self.assertEqual("body", rows[0].chunk_id)


if __name__ == "__main__":
    unittest.main()
