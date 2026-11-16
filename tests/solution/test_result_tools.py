"""Kiểm thử bổ sung cho công cụ xử lý artifact kết quả."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag.domain import ContractError
from agentic_rag.result_tools import merge_agent_runs


class ResultToolTests(unittest.TestCase):
    def test_repairs_replace_only_existing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            repair = root / "repair.json"
            output = root / "out.json"
            base.write_text(
                json.dumps(
                    {
                        "split": "dev",
                        "model": "m",
                        "elapsed_seconds": 1,
                        "traces": [{"query_id": "q1", "status": "error"}],
                    }
                ),
                encoding="utf-8",
            )
            repair.write_text(
                json.dumps(
                    {
                        "split": "dev",
                        "model": "m",
                        "elapsed_seconds": 2,
                        "traces": [{"query_id": "q1", "status": "answered"}],
                    }
                ),
                encoding="utf-8",
            )
            report = merge_agent_runs(base, [repair], output)
            self.assertEqual("answered", report["traces"][0]["status"])
            self.assertEqual(3.0, report["elapsed_seconds"])

    def test_unknown_repair_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            repair = root / "repair.json"
            base.write_text(
                json.dumps({"split": "dev", "model": "m", "traces": []}), encoding="utf-8"
            )
            repair.write_text(
                json.dumps(
                    {
                        "split": "dev",
                        "model": "m",
                        "traces": [{"query_id": "q1"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "absent from base"):
                merge_agent_runs(base, [repair], root / "out.json")


if __name__ == "__main__":
    unittest.main()
