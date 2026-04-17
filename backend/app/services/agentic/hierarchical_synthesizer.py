"""Hierarchical evidence synthesis for Agentic RAG final context."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.services.agentic.models import AgenticRetrievedChunk, SubQuerySummary
from app.services.agentic.prompts import HIERARCHICAL_SYNTHESIZER_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class HierarchicalSynthesizer:
    """Summarize selected chunks by sub-query and assemble compact context."""

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        timeout_seconds: float = 8.0,
        max_tokens: int = 900,
        max_chunks_per_summary: int = 4,
        max_excerpt_chars: int = 700,
        use_llm: bool = True,
    ) -> None:
        self._llm_provider = llm_provider
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tokens = max(256, int(max_tokens))
        self._max_chunks_per_summary = max(1, int(max_chunks_per_summary))
        self._max_excerpt_chars = max(120, int(max_excerpt_chars))
        self._use_llm = use_llm

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        config: Any | None = None,
    ) -> "HierarchicalSynthesizer":
        """Build synthesizer from app settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            max_chunks_per_summary=config.AGENTIC_MAX_CHUNKS_PER_SUBQUERY,
            use_llm=True,
        )

    async def summarize(
        self,
        *,
        chunks: list[AgenticRetrievedChunk],
        sub_queries: list[str],
    ) -> list[SubQuerySummary]:
        """Create one grounded summary per sub-query with supporting chunk IDs."""

        grouped = self._group_chunks(chunks=chunks, sub_queries=sub_queries)
        summaries: list[SubQuerySummary] = []
        for sub_query, group_chunks in grouped.items():
            if not group_chunks:
                continue
            selected_chunks = sorted(group_chunks, key=lambda chunk: (-chunk.score, chunk.chunk_id))[
                : self._max_chunks_per_summary
            ]
            if self._use_llm:
                try:
                    summary = await asyncio.wait_for(
                        self._summarize_with_llm(sub_query=sub_query, chunks=selected_chunks),
                        timeout=self._timeout_seconds,
                    )
                    summaries.append(
                        SubQuerySummary(
                            sub_query=sub_query,
                            summary=self._normalize_summary_text(summary),
                            supporting_chunk_ids=[chunk.chunk_id for chunk in selected_chunks],
                        )
                    )
                    continue
                except Exception as exc:
                    logger.info("HierarchicalSynthesizer fallback used for '%s': %s", sub_query, exc)

            summaries.append(self._fallback_summary(sub_query=sub_query, chunks=selected_chunks))

        return summaries

    def assemble(
        self,
        *,
        original_query: str,
        summaries: list[SubQuerySummary],
        chunks: list[AgenticRetrievedChunk],
    ) -> str:
        """Assemble final compact context for generation."""

        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        lines: list[str] = [
            f"Original question: {original_query.strip()}",
            "",
            "Sub-query summaries:",
        ]

        if summaries:
            for index, summary in enumerate(summaries, start=1):
                supporting_ids = [chunk_id for chunk_id in summary.supporting_chunk_ids if chunk_id in chunk_by_id]
                lines.append(f"[SQ{index}] {summary.sub_query}")
                lines.append(summary.summary.strip())
                lines.append(f"Supported by: {', '.join(supporting_ids) if supporting_ids else 'none'}")
                lines.append("")
        else:
            lines.append("No sub-query summaries were produced.")
            lines.append("")

        critical_evidence = self._critical_evidence(summaries=summaries, chunks=chunks)
        lines.append("Critical evidence:")
        if critical_evidence:
            for index, chunk in enumerate(critical_evidence, start=1):
                excerpt = self._excerpt(chunk.content, max_chars=350)
                lines.append(f"[{index}] chunk_id={chunk.chunk_id} source={chunk.source.value} score={chunk.score:.4f}")
                lines.append(excerpt)
        else:
            lines.append("No selected evidence chunks were provided.")

        return "\n".join(lines).strip()

    async def _summarize_with_llm(
        self,
        *,
        sub_query: str,
        chunks: list[AgenticRetrievedChunk],
    ) -> str:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        evidence = self._format_evidence(chunks)
        result = await provider.acomplete(
            [
                LLMMessage(
                    role="user",
                    content=(
                        f"Sub-query:\n{sub_query}\n\n"
                        "Summarize only the evidence below into clear, natural prose.\n\n"
                        f"Evidence:\n{evidence}"
                    ),
                )
            ],
            system_prompt=HIERARCHICAL_SYNTHESIZER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return result.content if isinstance(result, LLMResult) else str(result)

    def _group_chunks(
        self,
        *,
        chunks: list[AgenticRetrievedChunk],
        sub_queries: list[str],
    ) -> dict[str, list[AgenticRetrievedChunk]]:
        normalized_sub_queries = [item.strip() for item in sub_queries if item.strip()]
        if not normalized_sub_queries:
            normalized_sub_queries = ["selected evidence"]

        grouped: dict[str, list[AgenticRetrievedChunk]] = {sub_query: [] for sub_query in normalized_sub_queries}
        fallback_key = normalized_sub_queries[0]

        for chunk in chunks:
            matched = False
            covered_sub_queries = self._covered_sub_queries(chunk)
            for sub_query in normalized_sub_queries:
                if sub_query.casefold() in covered_sub_queries:
                    grouped[sub_query].append(chunk)
                    matched = True
            if not matched:
                grouped[fallback_key].append(chunk)

        return grouped

    @staticmethod
    def _covered_sub_queries(chunk: AgenticRetrievedChunk) -> set[str]:
        raw = chunk.metadata.get("covered_sub_queries") or chunk.metadata.get("covered_sub_query")
        values: set[str] = set()
        if isinstance(raw, str):
            values.add(raw.casefold())
        elif isinstance(raw, list):
            values.update(str(item).casefold() for item in raw if str(item).strip())
        return values

    def _fallback_summary(self, *, sub_query: str, chunks: list[AgenticRetrievedChunk]) -> SubQuerySummary:
        excerpts = [
            f"{chunk.chunk_id}: {self._excerpt(chunk.content, max_chars=self._max_excerpt_chars)}"
            for chunk in chunks
        ]
        return SubQuerySummary(
            sub_query=sub_query,
            summary=" ".join(excerpts).strip(),
            supporting_chunk_ids=[chunk.chunk_id for chunk in chunks],
        )

    @staticmethod
    def _normalize_summary_text(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            raise ValueError("LLM summary is empty")
        return normalized

    def _format_evidence(self, chunks: list[AgenticRetrievedChunk]) -> str:
        items: list[str] = []
        for chunk in chunks:
            items.append(
                f"chunk_id={chunk.chunk_id} source={chunk.source.value} score={chunk.score:.4f}\n"
                f"{self._excerpt(chunk.content, max_chars=self._max_excerpt_chars)}"
            )
        return "\n\n".join(items)

    @staticmethod
    def _critical_evidence(
        *,
        summaries: list[SubQuerySummary],
        chunks: list[AgenticRetrievedChunk],
    ) -> list[AgenticRetrievedChunk]:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        selected_ids: list[str] = []
        for summary in summaries:
            for chunk_id in summary.supporting_chunk_ids:
                if chunk_id not in selected_ids and chunk_id in chunk_by_id:
                    selected_ids.append(chunk_id)

        if not selected_ids:
            return sorted(chunks, key=lambda chunk: (-chunk.score, chunk.chunk_id))[:5]

        return [chunk_by_id[chunk_id] for chunk_id in selected_ids]

    @staticmethod
    def _excerpt(text: str, *, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."


__all__ = ["HierarchicalSynthesizer"]
