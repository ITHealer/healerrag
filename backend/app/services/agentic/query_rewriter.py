"""Query rewriting strategies for Agentic RAG retrieval retries."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.services.agentic.models import RewriteStrategy, RewrittenQuery
from app.services.agentic.prompts import QUERY_REWRITER_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Produce retrieval-oriented rewrite queries from missing evidence."""

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        timeout_seconds: float = 5.0,
        max_tokens: int = 900,
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
    ) -> "QueryRewriter":
        """Build rewriter from app settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            use_llm=True,
        )

    async def rewrite(
        self,
        *,
        original_query: str,
        missing_aspects: list[str],
        strategy: RewriteStrategy,
        iteration: int,
    ) -> RewrittenQuery:
        """Rewrite the query for one retrieval retry attempt."""

        normalized_query = original_query.strip()
        normalized_missing = self._dedupe(missing_aspects)
        normalized_iteration = max(1, int(iteration))

        if self._use_llm:
            try:
                rewritten = await asyncio.wait_for(
                    self._rewrite_with_llm(
                        original_query=normalized_query,
                        missing_aspects=normalized_missing,
                        strategy=strategy,
                        iteration=normalized_iteration,
                    ),
                    timeout=self._timeout_seconds,
                )
                return self._normalize_rewrite(
                    rewritten=rewritten,
                    original_query=normalized_query,
                    missing_aspects=normalized_missing,
                    strategy=strategy,
                    iteration=normalized_iteration,
                )
            except Exception as exc:
                logger.info("QueryRewriter fallback used: %s", exc)

        return self._fallback_rewrite(
            original_query=normalized_query,
            missing_aspects=normalized_missing,
            strategy=strategy,
            iteration=normalized_iteration,
        )

    async def _rewrite_with_llm(
        self,
        *,
        original_query: str,
        missing_aspects: list[str],
        strategy: RewriteStrategy,
        iteration: int,
    ) -> RewrittenQuery:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        user_content = (
            "Return only JSON. Do not wrap it in markdown.\n\n"
            f"Original query:\n{original_query}\n\n"
            f"Missing aspects:\n{json.dumps(missing_aspects, ensure_ascii=True)}\n\n"
            f"Strategy: {strategy.value}\n"
            f"Iteration: {iteration}"
        )
        result = await provider.acomplete(
            [LLMMessage(role="user", content=user_content)],
            system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        text = result.content if isinstance(result, LLMResult) else str(result)
        data = self._extract_json_object(text)
        data = self._coerce_rewrite_data(
            data=data,
            original_query=original_query,
            strategy=strategy,
            iteration=iteration,
        )
        return RewrittenQuery.model_validate(data)

    @staticmethod
    def _coerce_rewrite_data(
        *,
        data: dict[str, Any],
        original_query: str,
        strategy: RewriteStrategy,
        iteration: int,
    ) -> dict[str, Any]:
        coerced = dict(data)
        rewritten_query = (
            coerced.get("rewritten_query")
            or coerced.get("query")
            or coerced.get("rewrite")
            or coerced.get("RewrittenQuery")
            or coerced.get("rewrittenQuery")
        )
        if rewritten_query is not None:
            coerced["rewritten_query"] = str(rewritten_query)
        coerced.setdefault("original_query", original_query)
        coerced.setdefault("strategy", strategy.value)
        coerced.setdefault("iteration", iteration)
        return coerced

    def _normalize_rewrite(
        self,
        *,
        rewritten: RewrittenQuery,
        original_query: str,
        missing_aspects: list[str],
        strategy: RewriteStrategy,
        iteration: int,
    ) -> RewrittenQuery:
        rewritten_query = re.sub(r"\s+", " ", rewritten.rewritten_query).strip()
        if not rewritten_query:
            return self._fallback_rewrite(
                original_query=original_query,
                missing_aspects=missing_aspects,
                strategy=strategy,
                iteration=iteration,
            )

        return rewritten.model_copy(
            update={
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "strategy": strategy,
                "iteration": iteration,
            }
        )

    def _fallback_rewrite(
        self,
        *,
        original_query: str,
        missing_aspects: list[str],
        strategy: RewriteStrategy,
        iteration: int,
    ) -> RewrittenQuery:
        query = original_query or "Retrieve relevant evidence"
        missing_text = "; ".join(missing_aspects)

        if strategy == RewriteStrategy.EXPANSION:
            rewritten_query = self._expansion_query(query=query, missing_text=missing_text)
        elif strategy == RewriteStrategy.STEP_BACK:
            rewritten_query = self._step_back_query(query=query, missing_text=missing_text)
        elif strategy == RewriteStrategy.HYDE:
            rewritten_query = self._hyde_query(query=query, missing_text=missing_text)
        else:
            rewritten_query = self._expansion_query(query=query, missing_text=missing_text)

        return RewrittenQuery(
            original_query=query,
            rewritten_query=rewritten_query,
            strategy=strategy,
            iteration=iteration,
        )

    @staticmethod
    def _expansion_query(*, query: str, missing_text: str) -> str:
        if missing_text:
            return f"{query} missing evidence details: {missing_text}"
        return f"{query} related evidence details dates entities document sections"

    @staticmethod
    def _step_back_query(*, query: str, missing_text: str) -> str:
        if missing_text:
            return f"Broader background and document context for {query}: {missing_text}"
        return f"Broader background and document context for {query}"

    @staticmethod
    def _hyde_query(*, query: str, missing_text: str) -> str:
        if missing_text:
            return f"Hypothetical relevant passage for retrieval: {query}. It discusses {missing_text}."
        return f"Hypothetical relevant passage for retrieval: {query}."

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

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


__all__ = ["QueryRewriter"]
