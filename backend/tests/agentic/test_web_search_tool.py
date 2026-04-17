import unittest

from app.services.agentic.models import ChunkSource, WebSearchOutput, WebSearchResult, WebSearchSource
from app.services.agentic.web_search_tool import (
    WebSearchCredentialsError,
    WebSearchDisabledError,
    WebSearchTool,
)


class FakeOpenAIResponse:
    output_text = ""

    def __init__(self, raw: dict):
        self._raw = raw

    def model_dump(self, **_: object) -> dict:
        return self._raw


class WebSearchToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_search_blocks_before_backend_resolution(self) -> None:
        tool = WebSearchTool(
            enabled=False,
            backend="openai",
            openai_api_key="test-key",
        )

        with self.assertRaises(WebSearchDisabledError):
            await tool.search(query="latest NVDA price")

    def test_normalize_queries_dedupes_and_caps(self) -> None:
        tool = WebSearchTool(enabled=True, max_queries=2)

        queries = tool.normalize_queries(
            query="NVDA revenue",
            queries=["nvda revenue", "NVDA margin", "NVDA cash flow"],
        )

        self.assertEqual(queries, ["NVDA revenue", "NVDA margin"])

    def test_resolve_backend_prefers_google_in_auto_mode(self) -> None:
        tool = WebSearchTool(
            enabled=True,
            backend="auto",
            openai_api_key="openai-key",
            google_api_key="google-key",
        )

        self.assertEqual(tool.resolve_backend(), "google")

    def test_resolve_backend_reports_missing_credentials(self) -> None:
        tool = WebSearchTool(enabled=True, backend="google")

        with self.assertRaises(WebSearchCredentialsError):
            tool.resolve_backend()

    def test_extract_openai_text_and_sources(self) -> None:
        raw = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "text": "OpenAI answer",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "Example A",
                                    "start_index": 0,
                                    "end_index": 6,
                                },
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "Duplicate",
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        response = FakeOpenAIResponse(raw)

        text = WebSearchTool.extract_openai_text(response=response, raw=raw)
        sources = WebSearchTool.extract_openai_sources(raw)

        self.assertEqual(text, "OpenAI answer")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://example.com/a")
        self.assertEqual(sources[0].metadata["start_index"], 0)

    def test_extract_google_text_sources_and_queries(self) -> None:
        raw = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '[{"name":"google_search"}]'},
                            {"text": "Google grounded answer"},
                        ]
                    },
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/b", "title": "Example B"}},
                            {"web": {"uri": "https://example.com/b", "title": "Duplicate"}},
                        ],
                        "webSearchQueries": ["query b"],
                    },
                }
            ]
        }

        self.assertEqual(WebSearchTool.extract_google_text(raw), "Google grounded answer")
        self.assertEqual(WebSearchTool.extract_google_search_queries(raw), ["query b"])
        sources = WebSearchTool.extract_google_sources(raw)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://example.com/b")

    def test_google_http_options_sanitizes_versioned_base_url(self) -> None:
        official = WebSearchTool.build_google_http_options("https://generativelanguage.googleapis.com/v1beta")
        custom = WebSearchTool.build_google_http_options("https://proxy.local/v1")

        self.assertEqual(official, {"api_version": "v1beta"})
        self.assertEqual(custom, {"base_url": "https://proxy.local", "api_version": "v1"})

    def test_to_chunks_creates_stable_web_chunks(self) -> None:
        tool = WebSearchTool(enabled=True)
        output = WebSearchOutput(
            backend="google",
            model="gemini-3-flash-preview",
            results=[
                WebSearchResult(
                    query="NVDA revenue",
                    answer="NVDA reported revenue facts.",
                    sources=[
                        WebSearchSource(
                            url="https://example.com/nvda",
                            title="NVDA source",
                        )
                    ],
                    search_queries=["NVDA revenue"],
                    provider="google",
                    model="gemini-3-flash-preview",
                )
            ],
        )

        first = tool.to_chunks(output)
        second = tool.to_chunks(output)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].chunk_id, second[0].chunk_id)
        self.assertEqual(first[0].source, ChunkSource.WEB)
        self.assertEqual(first[0].metadata["web_url"], "https://example.com/nvda")
        self.assertEqual(first[0].metadata["covered_sub_queries"], ["NVDA revenue"])


if __name__ == "__main__":
    unittest.main()
