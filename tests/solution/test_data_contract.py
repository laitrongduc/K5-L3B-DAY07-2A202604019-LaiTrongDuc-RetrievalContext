"""Kiểm thử bổ sung cho contract dữ liệu Day 7."""

import unittest

from src.agentic_rag import ContractError, validate_dataset


def chunk(identifier: str = "c1", *, page: int = 1) -> dict[str, object]:
    return {
        "id": identifier,
        "text": "Yêu cầu hoàn tiền được xử lý trong 7 ngày làm việc.",
        "source_id": "policy",
        "source": "policy.pdf",
        "page": page,
        "metadata": {"version": "2025"},
    }


def answerable(identifier: str = "q1", *, split: str = "dev") -> dict[str, object]:
    return {
        "id": identifier,
        "split": split,
        "question": "Hoàn tiền mất bao lâu?",
        "expected_answer": "7 ngày làm việc",
        "gold": [{"source_id": "policy", "page": 1, "evidence": "trong 7 ngày làm việc"}],
    }


def unanswerable(identifier: str = "q2", *, split: str = "dev") -> dict[str, object]:
    return {
        "id": identifier,
        "split": split,
        "question": "Có hỗ trợ đổi chuyến bay không?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Corpus chỉ mô tả hoàn tiền.",
        "absence_terms": ["chuyến bay"],
    }


class DatasetContractTests(unittest.TestCase):
    def test_valid_dataset_reports_counts_without_gold(self) -> None:
        report = validate_dataset([chunk()], [answerable(), unanswerable()])
        self.assertEqual(report.chunks, 1)
        self.assertEqual(report.dev.answerable, 1)
        self.assertEqual(report.dev.unanswerable, 1)
        self.assertNotIn("gold", report.to_json())

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate id 'c1'"):
            validate_dataset([chunk(), chunk()], [answerable()])

    def test_unresolved_evidence_is_rejected(self) -> None:
        broken = answerable()
        broken["gold"] = [{"source_id": "policy", "page": 2, "evidence": "trong 7 ngày làm việc"}]
        with self.assertRaisesRegex(ContractError, "evidence does not resolve"):
            validate_dataset([chunk()], [broken])

    def test_evidence_may_resolve_across_exact_adjacent_spans(self) -> None:
        first = chunk("c1")
        first["text"] = "alpha beta "
        first["metadata"] = {"start_char": 0, "end_char": 11}
        second = chunk("c2")
        second["text"] = "gamma delta"
        second["metadata"] = {"start_char": 11, "end_char": 22}
        question = answerable()
        question["gold"] = [{"source_id": "policy", "page": 1, "evidence": "beta gamma"}]
        validate_dataset([first, second], [question])

    def test_unanswerable_requires_reason_and_expected_behavior(self) -> None:
        broken = unanswerable()
        broken.pop("absence_reason")
        with self.assertRaisesRegex(ContractError, "absence_reason"):
            validate_dataset([chunk()], [broken])

    def test_claimed_absence_term_must_not_appear_in_corpus(self) -> None:
        broken = unanswerable()
        broken["absence_terms"] = ["hoàn tiền"]
        with self.assertRaisesRegex(ContractError, "appears in chunk"):
            validate_dataset([chunk()], [broken])

    def test_minimum_counts_are_enforced_for_selected_split(self) -> None:
        with self.assertRaisesRegex(ContractError, "requires at least 15 answerable"):
            validate_dataset(
                [chunk()], [answerable(), unanswerable()], split="dev", require_minimums=True
            )

    def test_held_out_split_is_not_evidence_checked_during_dev(self) -> None:
        invalid_test = answerable("test-1", split="test")
        invalid_test["gold"] = [{"source_id": "missing", "page": 99, "evidence": "held out"}]
        report = validate_dataset([chunk()], [answerable(), invalid_test], split="dev")
        self.assertEqual(report.questions, 1)
        self.assertEqual(report.test.answerable, 0)


if __name__ == "__main__":
    unittest.main()
