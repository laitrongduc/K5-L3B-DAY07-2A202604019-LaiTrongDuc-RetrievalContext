from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agentic_rag.domain import ContractError
from agentic_rag.embeddings import EmbeddingConfig, OpenRouterEmbedder, load_embedding_config


class EmbeddingTests(unittest.TestCase):
    def test_load_config_and_embed_batches_in_response_index_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "https://example.test/api/v1/",
                        "model": "example/model",
                        "accepted_response_models": ["example/model"],
                        "dimensions": 2,
                        "batch_size": 2,
                        "timeout_seconds": 5,
                        "normalize": True,
                    }
                ),
                encoding="utf-8",
            )
            config = load_embedding_config(path)
        calls: list[dict[str, Any]] = []

        def transport(
            url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
        ) -> dict[str, Any]:
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            count = len(payload["input"])
            data = [{"index": index, "embedding": [3.0, 4.0]} for index in reversed(range(count))]
            return {
                "data": data,
                "model": "example/model",
                "usage": {"prompt_tokens": count, "total_tokens": count},
            }

        result = OpenRouterEmbedder(config, api_key="secret", transport=transport).embed(
            ["a", "b", "c"], input_type="search_document"
        )
        self.assertEqual(2, len(calls))
        self.assertNotIn("secret", json.dumps(calls[0]["payload"]))
        self.assertEqual((0.6, 0.8), result.vectors[0])
        self.assertTrue(
            all(math.isclose(math.dist(vector, (0.0, 0.0)), 1.0) for vector in result.vectors)
        )
        self.assertEqual(3, result.total_tokens)

    def test_rejects_wrong_dimension_non_finite_and_missing_key(self) -> None:
        config = EmbeddingConfig("https://example.test", "m", ("m",), 2, 2, 5.0, True)
        with self.assertRaisesRegex(ContractError, "OPENROUTER_API_KEY"):
            OpenRouterEmbedder(config, api_key="")

        responses = (
            {"data": [{"index": 0, "embedding": [1.0]}], "model": "m", "usage": {}},
            {"data": [{"index": 0, "embedding": [float("nan"), 1.0]}], "model": "m", "usage": {}},
        )
        for response in responses:
            with self.subTest(response=response):
                embedder = OpenRouterEmbedder(
                    config,
                    api_key="secret",
                    transport=lambda _url, _headers, _payload, _timeout, value=response: value,
                )
                with self.assertRaises(ContractError):
                    embedder.embed(["text"], input_type="search_query")

    def test_rejects_unexpected_response_model(self) -> None:
        config = EmbeddingConfig(
            "https://example.test", "expected/model", ("expected/model",), 2, 2, 5.0, True
        )
        embedder = OpenRouterEmbedder(
            config,
            api_key="secret",
            transport=lambda _url, _headers, _payload, _timeout: {
                "data": [{"index": 0, "embedding": [1.0, 1.0]}],
                "model": "other/model",
                "usage": {},
            },
        )
        with self.assertRaisesRegex(ContractError, "unexpected model"):
            embedder.embed(["text"], input_type="search_document")


if __name__ == "__main__":
    unittest.main()
