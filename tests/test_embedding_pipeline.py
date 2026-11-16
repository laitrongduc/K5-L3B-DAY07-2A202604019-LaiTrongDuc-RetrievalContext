from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_rag.domain import ContractError
from agentic_rag.embedding_pipeline import embed_corpus
from agentic_rag.embeddings import EmbeddingBatch


class EmbeddingPipelineTests(unittest.TestCase):
    def test_writes_checksum_manifest_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.json"
            config = root / "config.json"
            output = root / "output"
            chunks.write_text(
                json.dumps([{"id": "a", "text": "xin chào"}, {"id": "b", "text": "dữ liệu"}]),
                encoding="utf-8",
            )
            config.write_text(
                json.dumps(
                    {
                        "base_url": "https://example.test",
                        "model": "baai/bge-m3",
                        "accepted_response_models": ["BAAI/bge-m3"],
                        "dimensions": 2,
                        "batch_size": 2,
                        "timeout_seconds": 5,
                        "normalize": True,
                    }
                ),
                encoding="utf-8",
            )
            result = EmbeddingBatch(((1.0, 0.0), (0.0, 1.0)), ("BAAI/bge-m3",), 10, 10)
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret"}),
                patch(
                    "agentic_rag.embedding_pipeline.OpenRouterEmbedder.embed", return_value=result
                ),
            ):
                manifest = embed_corpus(chunks, config, output)
            self.assertEqual(2, manifest["vectors"])
            self.assertEqual(2, manifest["dimensions"])
            rows = json.loads((output / "embeddings.json").read_text(encoding="utf-8"))
            self.assertEqual(["a", "b"], [row["chunk_id"] for row in rows])
            with self.assertRaisesRegex(ContractError, "refusing to overwrite"):
                embed_corpus(chunks, config, output)


if __name__ == "__main__":
    unittest.main()
