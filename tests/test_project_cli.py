"""End-to-end behavior checks for the reusable project CLI."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        return subprocess.run(
            [sys.executable, "-m", "agentic_rag", *arguments],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_validate_example_data_outputs_stable_json(self) -> None:
        result = self.run_cli(
            "validate-data",
            "--chunks",
            "data/example_chunks.json",
            "--questions",
            "data/example_questions.json",
            "--split",
            "dev",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["chunks"], 1)
        self.assertEqual(report["dev"], {"answerable": 1, "unanswerable": 0})

    def test_minimum_failure_has_actionable_exit_code_and_message(self) -> None:
        result = self.run_cli(
            "validate-data",
            "--chunks",
            "data/example_chunks.json",
            "--questions",
            "data/example_questions.json",
            "--split",
            "dev",
            "--require-minimums",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires at least 15 answerable and 2 unanswerable", result.stderr)


if __name__ == "__main__":
    unittest.main()
