from __future__ import annotations

import unittest

from agentic_rag.controller import AgentConfig, OpenRouterController


class ControllerTests(unittest.TestCase):
    def test_controller_parses_action_and_records_usage_without_key(self) -> None:
        config = AgentConfig(
            "openai/gpt-4o-mini",
            "https://example.test",
            0,
            100,
            3,
            20,
            3,
            9000,
            "search_lexical",
            ("search_lexical",),
        )
        seen: dict[str, object] = {}

        def transport(url, headers, payload, timeout):
            seen.update({"headers": headers, "payload": payload})
            return {
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"action":"search","tool":"search_lexical","query":"q"}'
                        },
                    }
                ],
                "usage": {"total_tokens": 10},
            }

        controller = OpenRouterController(config, api_key="secret", transport=transport)
        action = controller.decide("question", [])
        self.assertEqual("search", action["action"])
        self.assertNotIn("secret", str(seen["payload"]))
        self.assertEqual(10, controller.calls[0]["total_tokens"])

    def test_controller_normalizes_narrow_tool_query_shape(self) -> None:
        config = AgentConfig(
            "m",
            "https://example.test",
            0,
            100,
            3,
            20,
            3,
            9000,
            "search_lexical",
            ("search_lexical",),
        )
        controller = OpenRouterController(
            config,
            api_key="secret",
            transport=lambda *_: {
                "model": "m",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"action":"search","tool/query":"blockchain"}'},
                    }
                ],
                "usage": {},
            },
        )
        action = controller.decide("q", [])
        self.assertEqual(
            {"action": "search", "tool": "search_lexical", "query": "blockchain"}, action
        )


if __name__ == "__main__":
    unittest.main()
