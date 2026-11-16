"""Kiểm thử bổ sung cho khâu chuẩn bị corpus PDF."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.agentic_rag import ContractError
from src.agentic_rag.ingestion import fixed_window_chunks, load_corpus_config, prepare_corpus


def config_payload(source_sha256: str) -> dict[str, object]:
    return {
        "document_id": "doc",
        "source_url": "https://example.test/doc.pdf",
        "source_sha256": source_sha256,
        "license": "Test license",
        "license_url": "https://example.test/license",
        "version": "1",
        "language": "en",
        "chunk_size_chars": 100,
        "chunk_overlap_chars": 20,
        "page_ranges": [{"start": 1, "end": 30, "chapter": 1, "chapter_title": "Test"}],
    }


class IngestionTests(unittest.TestCase):
    def test_fixed_windows_cover_source_without_gaps(self) -> None:
        text = "word " * 90
        chunks = fixed_window_chunks(text, size=100, overlap=20)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], len(text))
        self.assertTrue(all(value == text[start:end] for start, end, value in chunks))
        self.assertTrue(
            all(right[0] <= left[1] for left, right in zip(chunks, chunks[1:], strict=False))
        )

    def test_config_requires_at_least_thirty_unique_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            payload = config_payload("0" * 64)
            payload["page_ranges"] = [
                {"start": 1, "end": 29, "chapter": 1, "chapter_title": "Too short"}
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "at least 30 pages"):
                load_corpus_config(path)

    def test_prepare_corpus_uses_verified_pdf_and_writes_traceable_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "source.pdf"
            pdf.write_bytes(b"fixture-pdf")
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            config = root / "config.json"
            config.write_text(json.dumps(config_payload(digest)), encoding="utf-8")

            stats = prepare_corpus(
                pdf,
                config,
                root / "output",
                extractor=lambda _path, page: f"Page {page} " + ("content " * 30),
            )

            chunks = json.loads((root / "output/chunks.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["pages"], 30)
            self.assertGreater(stats["chunks"], 30)
            self.assertEqual(chunks[0]["metadata"]["start_char"], 0)
            self.assertEqual(chunks[0]["metadata"]["chapter"], 1)


if __name__ == "__main__":
    unittest.main()
