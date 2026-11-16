"""Kiểm thử bổ sung cho agent runtime của bài làm."""

from __future__ import annotations

import unittest

from agentic_rag.agent_runtime import parse_model_action, run_agent
from agentic_rag.domain import ContractError
from agentic_rag.retrieval import RetrievalResult


class StubController:
    def __init__(self, actions: list[dict[str, object]]) -> None:
        self.actions = iter(actions)

    def decide(self, question: str, history: list[dict[str, object]]) -> dict[str, object]:
        return next(self.actions)


class StubRetriever:
    def search(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        return [RetrievalResult("c1", "evidence", "source", 1, 1.0, 1, "bm25")]


class AgentRuntimeTests(unittest.TestCase):
    def test_search_then_grounded_answer(self) -> None:
        controller = StubController(
            [
                {"action": "search", "tool": "lexical", "query": "query"},
                {"action": "answer", "answer": "Đáp án [c1]", "citations": ["c1"]},
            ]
        )
        result = run_agent("question", controller, {"lexical": StubRetriever()})
        self.assertEqual("answered", result["status"])

    def test_invalid_citation_abstains_and_bad_json_fails(self) -> None:
        controller = StubController([{"action": "answer", "answer": "x", "citations": ["bad"]}])
        self.assertEqual(
            "abstain", run_agent("q", controller, {"lexical": StubRetriever()})["status"]
        )
        with self.assertRaisesRegex(ContractError, "valid JSON"):
            parse_model_action("not-json")

    def test_valid_citation_marker_is_rendered_deterministically(self) -> None:
        controller = StubController(
            [
                {"action": "search", "tool": "lexical", "query": "q"},
                {"action": "answer", "answer": "Đáp án", "citations": ["c1"]},
            ]
        )
        result = run_agent("q", controller, {"lexical": StubRetriever()})
        self.assertEqual("answered", result["status"])
        self.assertIn("[c1]", result["answer"])

    def test_unique_citation_prefix_is_resolved(self) -> None:
        controller = StubController(
            [
                {"action": "search", "tool": "lexical", "query": "q"},
                {"action": "answer", "answer": "Đáp án", "citations": ["c"]},
            ]
        )
        result = run_agent("q", controller, {"lexical": StubRetriever()})
        self.assertEqual(["c1"], result["citations"])

    def test_duplicate_search_abstains_without_calling_tool_twice(self) -> None:
        controller = StubController(
            [
                {"action": "search", "tool": "lexical", "query": " Query "},
                {"action": "search", "tool": "lexical", "query": "query"},
            ]
        )
        result = run_agent("q", controller, {"lexical": StubRetriever()})
        self.assertEqual("abstain", result["status"])
        self.assertEqual("duplicate_search", result["events"][-1]["event"])
        self.assertEqual(2, len(result["events"]))


if __name__ == "__main__":
    unittest.main()
