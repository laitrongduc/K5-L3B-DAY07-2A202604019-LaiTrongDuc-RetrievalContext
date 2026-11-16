"""Bounded agent state machine with validated model actions and citations."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from .domain import ContractError
from .retrieval import RetrievalResult, Retriever


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]: ...


def run_agent(
    question: str,
    controller: Controller,
    tools: dict[str, Retriever],
    *,
    max_searches: int = 3,
    top_k: int = 3,
    context_max_characters: int = 9000,
) -> dict[str, Any]:
    if not question.strip() or max_searches < 1 or top_k < 1 or context_max_characters < 1:
        raise ContractError("invalid agent question or positive limits")
    history: list[dict[str, Any]] = []
    context: dict[str, RetrievalResult] = {}
    events: list[dict[str, Any]] = []
    for _step in range(max_searches + 1):
        action = controller.decide(question, history)
        kind = action.get("action")
        if kind == "search":
            if len(events) >= max_searches:
                return _abstain(events, "search_limit")
            tool, query = action.get("tool"), action.get("query")
            if tool not in tools or not isinstance(query, str) or not query.strip():
                raise ContractError("model search requires registered tool and non-empty query")
            normalized_query = query.strip().casefold()
            if any(
                event.get("event") == "search"
                and event.get("tool") == tool
                and isinstance(event.get("query"), str)
                and str(event["query"]).strip().casefold() == normalized_query
                for event in events
            ):
                return _abstain(events, "duplicate_search")
            results = tools[tool].search(query, top_k=top_k)
            used = sum(len(result.text) for result in context.values())
            accepted: list[RetrievalResult] = []
            for result in results:
                if result.chunk_id in context:
                    continue
                if used + len(result.text) > context_max_characters:
                    break
                context[result.chunk_id] = result
                accepted.append(result)
                used += len(result.text)
            event = {
                "event": "search",
                "tool": tool,
                "query": query,
                "results": [asdict(result) for result in results],
                "context_ids": list(context),
            }
            events.append(event)
            history = [{"context": [asdict(result) for result in context.values()]}]
            continue
        if kind == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise ContractError("model answer requires text and citation list")
            resolved: list[str] = []
            for item in citations:
                if not isinstance(item, str):
                    resolved = []
                    break
                matches = [
                    chunk_id
                    for chunk_id in context
                    if chunk_id == item or chunk_id.startswith(item)
                ]
                if len(matches) != 1:
                    resolved = []
                    break
                resolved.append(matches[0])
            citations = resolved
            valid_ids = (
                bool(answer.strip())
                and bool(citations)
                and len(set(citations)) == len(citations)
                and all(isinstance(item, str) and item in context for item in citations)
            )
            if not valid_ids:
                return _abstain(events, "invalid_or_missing_citations")
            missing_markers = [item for item in citations if f"[{item}]" not in answer]
            if missing_markers:
                answer = answer.rstrip() + " " + " ".join(f"[{item}]" for item in missing_markers)
            return {
                "status": "answered",
                "answer": answer,
                "citations": citations,
                "events": events + [{"event": "answer", "citation_markers_added": missing_markers}],
            }
        if kind == "abstain":
            return _abstain(events, "model_abstain")
        raise ContractError("model action must be search, answer, or abstain")
    return _abstain(events, "step_limit")


def _abstain(events: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "status": "abstain",
        "answer": "Không đủ bằng chứng trong tài liệu để trả lời.",
        "citations": [],
        "events": events + [{"event": reason}],
    }


def parse_model_action(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ContractError(f"model action is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ContractError("model action must be a JSON object")
    return value
