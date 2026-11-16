"""Kiểm thử bổ sung cho pgvector store."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_rag.domain import ContractError
from agentic_rag.vector_store import load_vector_rows


class VectorStoreTests(unittest.TestCase):
    def test_load_vector_rows_requires_exact_ids_dimensions_and_norm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.json"
            embeddings = root / "embeddings.json"
            chunks.write_text(
                json.dumps(
                    [
                        {
                            "id": "chunk-1",
                            "source_id": "document",
                            "source": "https://example.test/document.pdf",
                            "page": 3,
                            "text": "nội dung",
                            "metadata": {"chapter": 1},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            embeddings.write_text(
                json.dumps([{"chunk_id": "chunk-1", "embedding": [0.6, 0.8]}]),
                encoding="utf-8",
            )
            rows = load_vector_rows(chunks, embeddings, 2)
            self.assertEqual("chunk-1", rows[0].chunk_id)
            self.assertEqual((0.6, 0.8), rows[0].embedding)

            embeddings.write_text(
                json.dumps([{"chunk_id": "chunk-1", "embedding": [1.0, 1.0]}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "normalized"):
                load_vector_rows(chunks, embeddings, 2)

    def test_rejects_missing_and_unknown_embedding_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.json"
            embeddings = root / "embeddings.json"
            chunks.write_text(
                json.dumps(
                    [
                        {
                            "id": "chunk-1",
                            "source_id": "document",
                            "source": "source",
                            "page": 1,
                            "text": "text",
                            "metadata": {},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            embeddings.write_text(
                json.dumps([{"chunk_id": "other", "embedding": [1.0, 0.0]}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "missing embedding"):
                load_vector_rows(chunks, embeddings, 2)


if __name__ == "__main__":
    unittest.main()
