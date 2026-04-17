"""Provider-backed web search for Agentic RAG.

The tool is disabled by default and performs no provider imports or remote calls
unless `enabled=True`. It normalizes provider responses into typed search output
and web evidence chunks for later retrieval orchestration.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

from app.services.agentic.models import (
    AgenticRetrievedChunk,
    ChunkSource,
    WebSearchOutput,
    WebSearchResult,
    WebSearchSource,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.core.config import Settings


class WebSearchError(RuntimeError):
    """Base exception for controlled web search failures."""


class WebSearchDisabledError(WebSearchError):
    """Raised when the tool is called while disabled by config."""


class WebSearchCredentialsError(WebSearchError):
    """Raised when the selected backend does not have credentials."""


class WebSearchDependencyError(WebSearchError):
    """Raised when a selected backend SDK is missing."""


class WebSearchTool:
    """Provider-backed web search tool with OpenAI and Google backends."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        backend: str = "auto",
        openai_api_key: str = "",
        openai_base_url: str = "",
        openai_model: str = "gpt-5.2",
        google_api_key: str = "",
        google_base_url: str = "",
        google_model: str = "gemini-3-flash-preview",
        timeout_seconds: float = 30.0,
        max_queries: int = 2,
        max_results: int = 5,
    ) -> None:
        self._enabled = enabled
        self._backend = backend.strip().lower() if backend else "auto"
        self._openai_api_key = openai_api_key.strip()
        self._openai_base_url = openai_base_url.strip()
        self._openai_model = openai_model.strip() or "gpt-5.2"
        self._google_api_key = google_api_key.strip()
        self._google_base_url = google_base_url.strip()
        self._google_model = google_model.strip() or "gemini-3-flash-preview"
        self._timeout_seconds = max(5.0, float(timeout_seconds))
        self._max_queries = max(1, int(max_queries))
        self._max_results = max(1, int(max_results))
        self._openai_client: Any | None = None
        self._google_client: Any | None = None

    @classmethod
    def from_settings(cls, config: "Settings | None" = None) -> "WebSearchTool":
        """Build a tool from application settings."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            enabled=config.AGENTIC_WEB_SEARCH_ENABLED,
            backend=config.WEB_SEARCH_TOOL_BACKEND,
            openai_api_key=config.OPENAI_API_KEY,
            openai_base_url=config.OPENAI_BASE_URL,
            openai_model=config.WEB_SEARCH_OPENAI_MODEL,
            google_api_key=config.GOOGLE_AI_API_KEY or config.GEMINI_API_KEY,
            google_base_url=config.GOOGLE_BASE_URL,
            google_model=config.WEB_SEARCH_GOOGLE_MODEL,
            timeout_seconds=config.WEB_SEARCH_TIMEOUT_SECONDS,
            max_queries=config.WEB_SEARCH_MAX_QUERIES,
            max_results=config.AGENTIC_WEB_SEARCH_MAX_RESULTS,
        )

    async def search(
        self,
        query: str | None = None,
        queries: list[str] | None = None,
        *,
        request_provider: str | None = None,
    ) -> WebSearchOutput:
        """Run provider-backed search and return normalized typed output."""

        if not self._enabled:
            raise WebSearchDisabledError("Web search is disabled. Set AGENTIC_WEB_SEARCH_ENABLED=true to enable it.")

        normalized_queries = self.normalize_queries(query=query, queries=queries)
        if not normalized_queries:
            raise ValueError("web search requires a non-empty query or queries")

        backend = self.resolve_backend(request_provider=request_provider)
        model = self.model_for_backend(backend)
        logger.info(
            "Agentic web search start backend=%s model=%s query_count=%s",
            backend,
            model,
            len(normalized_queries),
        )

        results: list[WebSearchResult] = []
        for index, normalized_query in enumerate(normalized_queries, start=1):
            if backend == "openai":
                result = await self._search_with_openai(normalized_query)
            elif backend == "google":
                result = await self._search_with_google(normalized_query)
            else:
                raise WebSearchError(f"Unsupported web search backend: {backend}")

            results.append(result)
            logger.info(
                "Agentic web search query done backend=%s index=%s sources=%s chars=%s",
                backend,
                index,
                len(result.sources),
                len(result.answer),
            )

        return WebSearchOutput(backend=backend, model=model, results=results)

    def normalize_queries(self, *, query: Any = None, queries: Any = None) -> list[str]:
        """Normalize, deduplicate, and cap query inputs."""

        normalized: list[str] = []
        if isinstance(query, str) and query.strip():
            normalized.append(query.strip())

        if isinstance(queries, list):
            for item in queries:
                if isinstance(item, str) and item.strip():
                    normalized.append(item.strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for item in normalized:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= self._max_queries:
                break

        return deduped

    def resolve_backend(self, *, request_provider: str | None = None) -> str:
        """Resolve effective backend and validate credentials."""

        normalized_provider = self._normalize_provider(request_provider)
        if normalized_provider:
            self._assert_backend_credentials(normalized_provider)
            return normalized_provider

        if self._backend in {"openai", "google"}:
            self._assert_backend_credentials(self._backend)
            return self._backend

        if self._backend != "auto":
            raise WebSearchError(f"Unsupported web search backend: {self._backend}")

        if self._google_api_key:
            return "google"
        if self._openai_api_key:
            return "openai"

        raise WebSearchCredentialsError(
            "Web search requires credentials. Set GOOGLE_AI_API_KEY/GEMINI_API_KEY or OPENAI_API_KEY."
        )

    def model_for_backend(self, backend: str) -> str:
        """Return configured model name for backend."""

        if backend == "openai":
            return self._openai_model
        if backend == "google":
            return self._google_model
        raise WebSearchError(f"Unsupported web search backend: {backend}")

    def to_chunks(self, output: WebSearchOutput) -> list[AgenticRetrievedChunk]:
        """Convert web search output into Agentic RAG evidence chunks."""

        chunks: list[AgenticRetrievedChunk] = []
        for result in output.results:
            content = result.answer.strip()
            if not content:
                continue

            primary_source = result.sources[0] if result.sources else None
            chunk_key = "|".join(
                [
                    result.provider,
                    result.model,
                    result.query,
                    primary_source.url if primary_source else "",
                    content,
                ]
            )
            chunk_id = f"web_{result.provider}_{self._stable_hash(chunk_key)}"
            chunks.append(
                AgenticRetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    score=1.0,
                    source=ChunkSource.WEB,
                    metadata={
                        "covered_sub_query": result.query,
                        "covered_sub_queries": [result.query],
                        "web_provider": result.provider,
                        "web_model": result.model,
                        "web_url": primary_source.url if primary_source else "",
                        "web_title": primary_source.title if primary_source else "",
                        "web_sources": [source.model_dump(mode="json") for source in result.sources[: self._max_results]],
                        "web_search_queries": result.search_queries,
                    },
                )
            )

        return chunks

    async def _search_with_openai(self, query: str) -> WebSearchResult:
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise WebSearchDependencyError(
                    "OpenAI web search requires the `openai` package to be installed."
                ) from exc

            client_kwargs: dict[str, Any] = {
                "api_key": self._openai_api_key,
                "timeout": self._timeout_seconds,
            }
            if self._openai_base_url:
                client_kwargs["base_url"] = self._openai_base_url
            self._openai_client = AsyncOpenAI(**client_kwargs)

        response = await self._openai_client.responses.create(
            model=self._openai_model,
            input=[
                {
                    "role": "user",
                    "content": f"Search the web and summarize factual updates for: {query}",
                }
            ],
            tools=[{"type": "web_search"}],
            temperature=0,
        )
        raw = response.model_dump(mode="json", warnings=False)
        text = self.extract_openai_text(response=response, raw=raw)
        return WebSearchResult(
            query=query,
            answer=text,
            sources=self.extract_openai_sources(raw)[: self._max_results],
            search_queries=[query],
            provider="openai",
            model=self._openai_model,
        )

    async def _search_with_google(self, query: str) -> WebSearchResult:
        if self._google_client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise WebSearchDependencyError(
                    "Google web search requires the `google-genai` package to be installed."
                ) from exc

            http_options = self.build_google_http_options(self._google_base_url)
            client_kwargs: dict[str, Any] = {"api_key": self._google_api_key}
            if http_options:
                client_kwargs["http_options"] = http_options
            self._google_client = genai.Client(**client_kwargs)

        try:
            from google.genai import types as gtypes
        except ImportError as exc:
            raise WebSearchDependencyError(
                "Google web search requires the `google-genai` package to expose google.genai.types."
            ) from exc

        response = await self._google_client.aio.models.generate_content(
            model=self._google_model,
            contents=query,
            config=gtypes.GenerateContentConfig(
                temperature=0,
                tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
            ),
        )
        raw = response.model_dump(exclude_none=True)
        return WebSearchResult(
            query=query,
            answer=self.extract_google_text(raw),
            sources=self.extract_google_sources(raw)[: self._max_results],
            search_queries=self.extract_google_search_queries(raw) or [query],
            provider="google",
            model=self._google_model,
        )

    def _assert_backend_credentials(self, backend: str) -> None:
        if backend == "openai" and not self._openai_api_key:
            raise WebSearchCredentialsError("WEB_SEARCH_TOOL_BACKEND=openai requires OPENAI_API_KEY.")
        if backend == "google" and not self._google_api_key:
            raise WebSearchCredentialsError("WEB_SEARCH_TOOL_BACKEND=google requires GOOGLE_AI_API_KEY or GEMINI_API_KEY.")

    @staticmethod
    def _normalize_provider(raw_value: str | None) -> str | None:
        if not isinstance(raw_value, str):
            return None
        normalized = raw_value.strip().lower()
        if normalized in {"openai", "google"}:
            return normalized
        return None

    @staticmethod
    def _stable_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def extract_openai_text(*, response: Any, raw: dict[str, Any]) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: list[str] = []
        for item in raw.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)

        return "".join(chunks).strip()

    @staticmethod
    def extract_openai_sources(raw: dict[str, Any]) -> list[WebSearchSource]:
        sources: list[WebSearchSource] = []
        seen: set[str] = set()
        for item in raw.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                for annotation in content.get("annotations", []):
                    if not isinstance(annotation, dict):
                        continue
                    annotation_type = annotation.get("type")
                    if annotation_type not in {"url_citation", None, ""}:
                        continue
                    url = annotation.get("url")
                    if not isinstance(url, str) or not url.startswith("http"):
                        continue
                    if url in seen:
                        continue
                    seen.add(url)
                    metadata: dict[str, Any] = {}
                    for key in ("start_index", "end_index"):
                        if isinstance(annotation.get(key), int):
                            metadata[key] = annotation[key]
                    sources.append(
                        WebSearchSource(
                            url=url,
                            title=annotation.get("title") or url,
                            metadata=metadata,
                        )
                    )
        return sources

    @classmethod
    def extract_google_text(cls, raw: dict[str, Any]) -> str:
        chunks: list[str] = []
        for candidate in raw.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        return cls.strip_tool_json_prefix("".join(chunks).strip())

    @staticmethod
    def extract_google_sources(raw: dict[str, Any]) -> list[WebSearchSource]:
        sources: list[WebSearchSource] = []
        seen: set[str] = set()
        for candidate in raw.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            grounding = candidate.get("grounding_metadata") or candidate.get("groundingMetadata")
            if not isinstance(grounding, dict):
                continue
            chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
            for source in chunks:
                if not isinstance(source, dict):
                    continue
                web = source.get("web")
                if not isinstance(web, dict):
                    continue
                uri = web.get("uri")
                if not isinstance(uri, str) or not uri.startswith("http"):
                    continue
                if uri in seen:
                    continue
                seen.add(uri)
                sources.append(
                    WebSearchSource(
                        url=uri,
                        title=web.get("title") or uri,
                    )
                )
        return sources

    @staticmethod
    def extract_google_search_queries(raw: dict[str, Any]) -> list[str]:
        for candidate in raw.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            grounding = candidate.get("grounding_metadata") or candidate.get("groundingMetadata")
            if not isinstance(grounding, dict):
                continue
            queries = grounding.get("web_search_queries") or grounding.get("webSearchQueries")
            if not isinstance(queries, list):
                continue
            return [query.strip() for query in queries if isinstance(query, str) and query.strip()]
        return []

    @staticmethod
    def strip_tool_json_prefix(text: str) -> str:
        normalized = text.lstrip()
        if not normalized.startswith("["):
            return text

        decoder = json.JSONDecoder()
        try:
            parsed, end_index = decoder.raw_decode(normalized)
        except json.JSONDecodeError:
            return text

        if not isinstance(parsed, list) or not parsed:
            return text
        if not all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in parsed):
            return text

        return normalized[end_index:].lstrip()

    @staticmethod
    def build_google_http_options(base_url: str | None) -> dict[str, Any] | None:
        if not base_url:
            return None

        stripped = base_url.strip()
        if not stripped:
            return None

        parsed = urlparse(stripped)
        hostname = (parsed.hostname or "").lower()
        path = (parsed.path or "").rstrip("/")

        is_official_endpoint = "generativelanguage.googleapis.com" in hostname
        if is_official_endpoint and path in {"/v1", "/v1beta"}:
            return {"api_version": path.lstrip("/")}

        if path in {"/v1", "/v1beta"}:
            sanitized_url = urlunparse(parsed._replace(path="", params="", query="", fragment=""))
            return {"base_url": sanitized_url, "api_version": path.lstrip("/")}

        return {"base_url": stripped}


__all__ = [
    "WebSearchCredentialsError",
    "WebSearchDependencyError",
    "WebSearchDisabledError",
    "WebSearchError",
    "WebSearchTool",
]
