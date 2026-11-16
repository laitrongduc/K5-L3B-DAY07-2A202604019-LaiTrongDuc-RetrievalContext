"""Tests for immutable benchmark manifests."""

import json
import tempfile
import unittest
from pathlib import Path

from src.agentic_rag import ContractError
from src.agentic_rag.freezing import freeze_benchmark, verify_benchmark_freeze


class FreezeTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, Path]:
        chunks = root / "chunks.json"
        questions = root / "questions.json"
        chunks.write_text(
            json.dumps(
                [
                    {
                        "id": "c1",
                        "text": "evidence",
                        "source_id": "doc",
                        "source": "doc.pdf",
                        "page": 1,
                        "metadata": {},
                    }
                ]
            ),
            encoding="utf-8",
        )
        rows = []
        for split, count in (("dev", 15), ("test", 5)):
            for index in range(count):
                rows.append(
                    {
                        "id": f"{split}-{index}",
                        "split": split,
                        "question": "Question?",
                        "expected_answer": "Answer",
                        "gold": [{"source_id": "doc", "page": 1, "evidence": "evidence"}],
                    }
                )
            for index in range(2):
                rows.append(
                    {
                        "id": f"{split}-u-{index}",
                        "split": split,
                        "question": "Unknown?",
                        "answerable": False,
                        "expected_behavior": "search_and_abstain",
                        "absence_reason": "Checked corpus.",
                    }
                )
        questions.write_text(json.dumps(rows), encoding="utf-8")
        return chunks, questions

    def test_freeze_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, questions = self._dataset(root)
            manifest_path = root / "freeze.json"
            created = freeze_benchmark(chunks, questions, manifest_path)
            verified = verify_benchmark_freeze(manifest_path)
            self.assertEqual(created, verified)
            self.assertEqual(verified["counts"]["questions"], 24)

    def test_modified_frozen_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, questions = self._dataset(root)
            manifest_path = root / "freeze.json"
            freeze_benchmark(chunks, questions, manifest_path)
            questions.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "checksum mismatch"):
                verify_benchmark_freeze(manifest_path)

    def test_freeze_resolves_inputs_outside_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            benchmark = root / "benchmark"
            data.mkdir()
            benchmark.mkdir()
            chunks, questions = self._dataset(data)
            manifest_path = benchmark / "freeze.json"
            manifest = freeze_benchmark(chunks, questions, manifest_path)
            self.assertEqual("../data/chunks.json", manifest["chunks"]["path"])
            self.assertEqual(manifest, verify_benchmark_freeze(manifest_path))


if __name__ == "__main__":
    unittest.main()
