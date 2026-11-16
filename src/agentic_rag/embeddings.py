"""Validated OpenRouter embedding adapter with deterministic local normalization."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import ContractError

JsonTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    base_url: str
    model: str
    accepted_response_models: tuple[str, ...]
    dimensions: int
    batch_size: int
    timeout_seconds: float
    normalize: bool


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    response_models: tuple[str, ...]
    prompt_tokens: int
    total_tokens: int


def load_embedding_config(path: Path) -> EmbeddingConfig:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: invalid embedding config: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")

    def text_field(name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise ContractError(f"embedding config {name}: expected non-empty string")
        return item.rstrip("/") if name == "base_url" else item.strip()

    dimensions = value.get("dimensions")
    batch_size = value.get("batch_size")
    timeout = value.get("timeout_seconds")
    normalize = value.get("normalize")
    if type(dimensions) is not int or dimensions < 1:
        raise ContractError("embedding config dimensions: expected positive integer")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ContractError("embedding config batch_size: expected integer in [1, 128]")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ContractError("embedding config timeout_seconds: expected positive number")
    if type(normalize) is not bool:
        raise ContractError("embedding config normalize: expected boolean")
    accepted = value.get("accepted_response_models", [text_field("model")])
    if (
        not isinstance(accepted, list)
        or not accepted
        or any(not isinstance(model, str) or not model.strip() for model in accepted)
    ):
        raise ContractError("embedding config accepted_response_models: expected non-empty strings")
    accepted_models = tuple(model.strip() for model in accepted)
    if len(set(model.casefold() for model in accepted_models)) != len(accepted_models):
        raise ContractError("embedding config accepted_response_models: duplicate model alias")
    return EmbeddingConfig(
        base_url=text_field("base_url"),
        model=text_field("model"),
        accepted_response_models=accepted_models,
        dimensions=dimensions,
        batch_size=batch_size,
        timeout_seconds=float(timeout),
        normalize=normalize,
    )


def _http_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ContractError(f"OpenRouter embedding HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ContractError(f"OpenRouter embedding request failed: {error.reason}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise ContractError("OpenRouter embedding response was not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractError("OpenRouter embedding response must be an object")
    return value


def _validated_vectors(
    response: dict[str, Any], *, expected_count: int, dimensions: int, normalize: bool
) -> tuple[tuple[float, ...], ...]:
    data = response.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise ContractError(
            f"embedding response count mismatch: expected {expected_count}, found "
            f"{len(data) if isinstance(data, list) else 'non-list'}"
        )
    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for item in data:
        if not isinstance(item, dict):
            raise ContractError("embedding response item must be an object")
        index, raw_vector = item.get("index"), item.get("embedding")
        if type(index) is not int or not 0 <= index < expected_count or ordered[index] is not None:
            raise ContractError(f"embedding response has invalid or duplicate index: {index!r}")
        if not isinstance(raw_vector, list) or len(raw_vector) != dimensions:
            raise ContractError(
                f"embedding {index} dimension mismatch: expected {dimensions}, found "
                f"{len(raw_vector) if isinstance(raw_vector, list) else 'non-list'}"
            )
        if any(
            not isinstance(number, (int, float)) or not math.isfinite(number)
            for number in raw_vector
        ):
            raise ContractError(f"embedding {index} contains a non-finite/non-numeric value")
        vector = tuple(float(number) for number in raw_vector)
        norm = math.sqrt(sum(number * number for number in vector))
        if norm <= 1e-12:
            raise ContractError(f"embedding {index} has zero norm")
        ordered[index] = tuple(number / norm for number in vector) if normalize else vector
    if any(vector is None for vector in ordered):
        raise ContractError("embedding response omitted one or more indices")
    return tuple(vector for vector in ordered if vector is not None)


class OpenRouterEmbedder:
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        api_key: str | None = None,
        transport: JsonTransport = _http_json,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        if not key.strip():
            raise ContractError("OPENROUTER_API_KEY is required for real embeddings")
        self._config = config
        self._api_key = key.strip()
        self._transport = transport

    def embed(self, texts: Sequence[str], *, input_type: str) -> EmbeddingBatch:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ContractError("embedding input must contain non-empty strings")
        if input_type not in {"search_document", "search_query"}:
            raise ContractError("input_type must be search_document or search_query")
        all_vectors: list[tuple[float, ...]] = []
        prompt_tokens = 0
        total_tokens = 0
        observed_models: list[str] = []
        accepted_models = {model.casefold() for model in self._config.accepted_response_models}
        for start in range(0, len(texts), self._config.batch_size):
            batch = list(texts[start : start + self._config.batch_size])
            response = self._transport(
                f"{self._config.base_url}/embeddings",
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                {"model": self._config.model, "input": batch, "input_type": input_type},
                self._config.timeout_seconds,
            )
            model = response.get("model")
            if not isinstance(model, str) or not model:
                raise ContractError("embedding response did not identify the model")
            if model.casefold() not in accepted_models:
                raise ContractError(
                    f"embedding provider returned unexpected model: accepted "
                    f"{self._config.accepted_response_models}, received {model}"
                )
            if model not in observed_models:
                observed_models.append(model)
            all_vectors.extend(
                _validated_vectors(
                    response,
                    expected_count=len(batch),
                    dimensions=self._config.dimensions,
                    normalize=self._config.normalize,
                )
            )
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                raise ContractError("embedding response usage must be an object")
            for name in ("prompt_tokens", "total_tokens"):
                amount = usage.get(name, 0)
                if type(amount) is not int or amount < 0:
                    raise ContractError(f"embedding usage {name} must be a non-negative integer")
            prompt_tokens += usage.get("prompt_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
        if not observed_models:
            raise AssertionError("embedding request completed without a response model")
        return EmbeddingBatch(
            tuple(all_vectors), tuple(observed_models), prompt_tokens, total_tokens
        )
