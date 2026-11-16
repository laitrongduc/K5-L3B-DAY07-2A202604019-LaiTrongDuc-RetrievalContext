"""Behavior checks for the deterministic quality orchestrator."""

import sys
import tempfile
import unittest
from pathlib import Path

from src.agentic_rag.quality import quality_commands, run_quality_checks


class QualityTests(unittest.TestCase):
    def test_pipeline_contains_all_required_gate_types(self) -> None:
        rendered = [" ".join(command) for command in quality_commands()]
        self.assertTrue(any("ruff format --check" in command for command in rendered))
        self.assertTrue(any("ruff check" in command for command in rendered))
        self.assertTrue(any("mypy" in command for command in rendered))
        self.assertTrue(any("compileall" in command for command in rendered))
        self.assertTrue(any("unittest discover" in command for command in rendered))

    def test_pipeline_stops_at_first_failure(self) -> None:
        commands = (
            (sys.executable, "-c", "raise SystemExit(7)"),
            (sys.executable, "-c", "raise SystemExit(0)"),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_quality_checks(Path(directory), commands)
        self.assertEqual(result, 7)


if __name__ == "__main__":
    unittest.main()
