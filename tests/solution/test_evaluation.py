"""Kiểm thử bổ sung cho retrieval evaluation."""

from __future__ import annotations

import unittest

from agentic_rag.domain import BenchmarkQuestion
from agentic_rag.evaluation import _reference_aware, _summarize
from agentic_rag.retrieval import RetrievalResult


class EvaluationTests(unittest.TestCase):
    def test_summary_reports_counts_rates_and_trace_without_gold_text(self) -> None:
        questions = [
            BenchmarkQuestion("q1", "dev", "Q?", True, "A", (), None, (), None),
            BenchmarkQuestion("q2", "dev", "Q?", True, "A", (), None, (), None),
        ]
        result = RetrievalResult("gold", "text", "source", 1, 0.9, 1, "method")
        miss = RetrievalResult("miss", "text", "source", 2, 0.8, 1, "method")
        summary = _summarize(
            "method",
            questions,
            {"q1": [result], "q2": [miss]},
            {"q1": {"gold"}, "q2": {"gold-2"}},
            elapsed_seconds=0.5,
        )
        self.assertEqual({"n": 1, "N": 2, "rate": 0.5}, summary["hit_at_1"])
        self.assertNotIn("gold", summary["traces"][0])

    def test_reference_aware_prefers_body_caption_over_toc(self) -> None:
        toc = RetrievalResult("toc", "Hình vẽ 3: mục lục", "s", 6, 2.0, 1, "bm25")
        body = RetrievalResult("body", "Hình vẽ 3: nội dung", "s", 25, 1.0, 2, "bm25")
        results = _reference_aware("Theo Hình 3, nội dung gì?", [toc, body], {"toc": 0, "body": 3})
        self.assertEqual("body", results[0].chunk_id)
        self.assertEqual(1, results[0].rank)


if __name__ == "__main__":
    unittest.main()
