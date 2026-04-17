"""Query analysis and decomposition for Agentic RAG."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.services.agentic.models import QueryAnalysisResult, QueryComplexity
from app.services.agentic.prompts import QUERY_ANALYZER_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Classify user query complexity and produce retrieval sub-queries."""

    _VI_HINTS = {
        "cach",
        "c\u00e1ch",
        "gi\u00fap",
        "h\u1ecfi",
        "l\u00e0 g\u00ec",
        "nh\u01b0 th\u1ebf n\u00e0o",
        "\u0111\u0103ng nh\u1eadp",
        "m\u1eadt kh\u1ea9u",
        "t\u00e0i li\u1ec7u",
        "ph\u00e2n t\u00edch",
        "so s\u00e1nh",
        "v\u00e0",
    }
    _NO_RETRIEVAL_PHRASES = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "xin ch\u00e0o",
        "ch\u00e0o",
        "c\u1ea3m \u01a1n",
        "ok",
    }
    _MULTI_HINTS = {
        "compare",
        "comparison",
        "analyze",
        "analysis",
        "trend",
        "over time",
        "multi",
        "ph\u00e2n t\u00edch",
        "so s\u00e1nh",
        "xu h\u01b0\u1edbng",
        "theo th\u1eddi gian",
        "10 n\u0103m",
        "nhi\u1ec1u",
    }

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        timeout_seconds: float = 5.0,
        max_tokens: int = 1200,
        use_llm: bool = True,
    ) -> None:
        self._llm_provider = llm_provider
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tokens = max(256, int(max_tokens))
        self._use_llm = use_llm

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        config: Any | None = None,
    ) -> "QueryAnalyzer":
        """Build analyzer from app settings without importing settings at module load."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            use_llm=True,
        )

    async def analyze(
        self,
        query: str,
        history: list[dict[str, Any]] | None = None,
    ) -> QueryAnalysisResult:
        """Analyze a query with LLM JSON output and rule-based fallback."""

        normalized_query = query.strip()
        if not normalized_query:
            return self._fallback_analysis(query)

        if self._use_llm:
            try:
                return await asyncio.wait_for(
                    self._analyze_with_llm(normalized_query, history or []),
                    timeout=self._timeout_seconds,
                )
            except Exception as exc:
                logger.info("QueryAnalyzer fallback used: %s", exc)

        return self._fallback_analysis(normalized_query)

    async def _analyze_with_llm(
        self,
        query: str,
        history: list[dict[str, Any]],
    ) -> QueryAnalysisResult:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        history_preview = self._history_preview(history)
        user_content = (
            "Return only JSON. Do not wrap it in markdown.\n\n"
            f"User query:\n{query}\n\n"
            f"Recent history:\n{history_preview or 'None'}"
        )
        result = await provider.acomplete(
            [LLMMessage(role="user", content=user_content)],
            system_prompt=QUERY_ANALYZER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        text = result.content if isinstance(result, LLMResult) else str(result)
        data = self._extract_json_object(text)
        data = self._coerce_analysis_data(data, query)
        return QueryAnalysisResult.model_validate(data)

    @staticmethod
    def _coerce_analysis_data(data: dict[str, Any], query: str) -> dict[str, Any]:
        coerced = dict(data)
        complexity = str(coerced.get("complexity", "")).strip().lower()
        aliases = {
            "none": QueryComplexity.NO_RETRIEVAL.value,
            "no": QueryComplexity.NO_RETRIEVAL.value,
            "no retrieval": QueryComplexity.NO_RETRIEVAL.value,
            "simple": QueryComplexity.SINGLE_HOP.value,
            "low": QueryComplexity.SINGLE_HOP.value,
            "medium": QueryComplexity.SINGLE_HOP.value,
            "single": QueryComplexity.SINGLE_HOP.value,
            "single-hop": QueryComplexity.SINGLE_HOP.value,
            "complex": QueryComplexity.MULTI_HOP.value,
            "high": QueryComplexity.MULTI_HOP.value,
            "multi": QueryComplexity.MULTI_HOP.value,
            "multi-hop": QueryComplexity.MULTI_HOP.value,
        }
        if complexity in aliases:
            coerced["complexity"] = aliases[complexity]

        sub_queries = coerced.get("sub_queries")
        if isinstance(sub_queries, str):
            coerced["sub_queries"] = [sub_queries]
        elif not isinstance(sub_queries, list):
            coerced["sub_queries"] = []

        if coerced.get("complexity") != QueryComplexity.NO_RETRIEVAL.value and not coerced["sub_queries"]:
            coerced["sub_queries"] = [query]

        entities = coerced.get("entities")
        if isinstance(entities, str):
            coerced["entities"] = [entities]
        elif not isinstance(entities, list):
            coerced["entities"] = []
        return coerced

    def _fallback_analysis(self, query: str) -> QueryAnalysisResult:
        normalized = query.strip()
        language = self._detect_language(normalized)
        if self._is_no_retrieval(normalized):
            return QueryAnalysisResult(
                complexity=QueryComplexity.NO_RETRIEVAL,
                sub_queries=[],
                entities=[],
                temporal_range=None,
                language=language,
                strategy_hint="no_retrieval",
                reasoning="Rule-based fallback classified the query as greeting or short non-retrieval text.",
            )

        sub_queries = self._split_sub_queries(normalized)
        temporal_range = self._extract_temporal_range(normalized)
        is_multi = len(sub_queries) > 1 or self._has_multi_hop_signals(normalized, temporal_range)
        complexity = QueryComplexity.MULTI_HOP if is_multi else QueryComplexity.SINGLE_HOP

        if not sub_queries:
            sub_queries = [normalized]

        if complexity == QueryComplexity.SINGLE_HOP:
            sub_queries = [normalized]

        return QueryAnalysisResult(
            complexity=complexity,
            sub_queries=sub_queries,
            entities=self._extract_entities(normalized),
            temporal_range=temporal_range,
            language=language,
            strategy_hint="multi_request" if len(sub_queries) > 1 else "direct",
            reasoning="Rule-based fallback used lexical signals and simple decomposition.",
        )

    @classmethod
    def _is_no_retrieval(cls, query: str) -> bool:
        normalized = query.strip().casefold()
        if normalized in cls._NO_RETRIEVAL_PHRASES:
            return True
        if len(normalized.split()) <= 3 and any(phrase in normalized for phrase in cls._NO_RETRIEVAL_PHRASES):
            return True
        return False

    @classmethod
    def _has_multi_hop_signals(cls, query: str, temporal_range: str | None) -> bool:
        normalized = query.casefold()
        if temporal_range:
            return True
        if any(hint in normalized for hint in cls._MULTI_HINTS):
            return True
        if re.search(r"\b(and|vs|versus)\b", normalized):
            return True
        if " v\u00e0 " in normalized or ";" in normalized:
            return True
        return False

    @classmethod
    def _split_sub_queries(cls, query: str) -> list[str]:
        normalized = query.strip()
        separators = r"\s*(?:;|\n|\bv\u00e0\b|\band\b|,)\s*"
        parts = [part.strip(" .?") for part in re.split(separators, normalized, flags=re.IGNORECASE)]
        parts = [part for part in parts if len(part.split()) >= 2]
        if len(parts) <= 1:
            return [normalized]
        if len(parts) > 6:
            return [normalized]
        return parts

    @classmethod
    def _detect_language(cls, query: str) -> str:
        if re.search(r"[\u4e00-\u9fff]", query):
            return "zh"
        normalized = query.casefold()
        if re.search(
            r"[\u00e0\u00e1\u1ea1\u1ea3\u00e3\u00e2\u1ea7\u1ea5\u1ead\u1ea9\u1eab"
            r"\u0103\u1eb1\u1eaf\u1eb7\u1eb3\u1eb5\u00e8\u00e9\u1eb9\u1ebb"
            r"\u1ebd\u00ea\u1ec1\u1ebf\u1ec7\u1ec3\u1ec5\u00ec\u00ed\u1ecb"
            r"\u1ec9\u0129\u00f2\u00f3\u1ecd\u1ecf\u00f5\u00f4\u1ed3\u1ed1"
            r"\u1ed9\u1ed5\u1ed7\u01a1\u1edd\u1edb\u1ee3\u1edf\u1ee1\u00f9"
            r"\u00fa\u1ee5\u1ee7\u0169\u01b0\u1eeb\u1ee9\u1ef1\u1eed\u1eef"
            r"\u1ef3\u00fd\u1ef5\u1ef7\u1ef9\u0111]",
            normalized,
        ):
            return "vi"
        if any(hint in normalized for hint in cls._VI_HINTS):
            return "vi"
        return "en"

    @staticmethod
    def _extract_temporal_range(query: str) -> str | None:
        years = re.findall(r"\b(?:19|20)\d{2}\b", query)
        if len(years) >= 2:
            return f"{years[0]}-{years[-1]}"
        if years:
            return years[0]
        match = re.search(r"\b(\d+)\s*(?:years|n\u0103m)\b", query, flags=re.IGNORECASE)
        if match:
            return f"last_{match.group(1)}_years"
        return None

    @staticmethod
    def _extract_entities(query: str) -> list[str]:
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9_.-]{1,}\b", query)
        candidates.extend(re.findall(r"\b[A-Z]{2,}\b", query))
        seen: set[str] = set()
        entities: list[str] = []
        for candidate in candidates:
            if candidate.casefold() in {"how", "what", "when", "where", "why"}:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            entities.append(candidate)
        return entities[:10]

    @staticmethod
    def _history_preview(history: list[dict[str, Any]]) -> str:
        preview = []
        for item in history[-4:]:
            role = item.get("role", "unknown")
            content = str(item.get("content", ""))[:300]
            preview.append(f"{role}: {content}")
        return "\n".join(preview)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")

        parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON output is not an object")
        return parsed


__all__ = ["QueryAnalyzer"]
