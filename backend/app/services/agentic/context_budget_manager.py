"""Deterministic final-context selection for Agentic RAG."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.agentic.models import AgenticRetrievedChunk, ChunkSource, ContextBudgetDecision

if TYPE_CHECKING:
    from app.core.config import Settings


@dataclass(frozen=True)
class _Candidate:
    chunk: AgenticRetrievedChunk
    token_count: int
    sub_queries: tuple[str, ...]


class ContextBudgetManager:
    """Select a compact, coverage-aware set of chunks for final generation.

    This component is intentionally deterministic and does not call an LLM.
    """

    def __init__(
        self,
        *,
        max_final_context_tokens: int = 5000,
        max_final_chunks: int = 8,
        max_chunks_per_subquery: int = 2,
    ) -> None:
        self._max_final_context_tokens = max(1, int(max_final_context_tokens))
        self._max_final_chunks = max(1, int(max_final_chunks))
        self._max_chunks_per_subquery = max(1, int(max_chunks_per_subquery))

    @classmethod
    def from_settings(cls, config: "Settings | None" = None) -> "ContextBudgetManager":
        """Build a manager from application settings."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            max_final_context_tokens=config.AGENTIC_MAX_FINAL_CONTEXT_TOKENS,
            max_final_chunks=config.AGENTIC_MAX_FINAL_CHUNKS,
            max_chunks_per_subquery=config.AGENTIC_MAX_CHUNKS_PER_SUBQUERY,
        )

    def select(
        self,
        *,
        chunks: list[AgenticRetrievedChunk],
        sub_queries: list[str],
    ) -> ContextBudgetDecision:
        """Select chunks while preserving per-sub-query coverage when possible."""

        candidates = self._dedupe_chunks(chunks)
        normalized_sub_queries = self._normalize_sub_queries(sub_queries)
        selected: list[_Candidate] = []
        selected_ids: set[str] = set()

        grouped = self._group_by_sub_query(candidates, normalized_sub_queries)

        # Pass 1: keep the strongest available chunk for each requested sub-query.
        for sub_query in normalized_sub_queries:
            for candidate in grouped.get(sub_query, []):
                if self._can_add(candidate, selected, selected_ids):
                    self._add_candidate(candidate, selected, selected_ids)
                    break

        # Pass 2: fill remaining chunk slots by score, source diversity, then ID.
        for candidate in self._sort_for_fill(candidates, selected):
            if self._can_add(candidate, selected, selected_ids):
                self._add_candidate(candidate, selected, selected_ids)

        # Pass 3: enforce token budget deterministically.
        dropped_for_budget: list[_Candidate] = []
        while self._total_tokens(selected) > self._max_final_context_tokens and selected:
            removable = self._choose_budget_drop(selected, normalized_sub_queries)
            selected.remove(removable)
            selected_ids.remove(removable.chunk.chunk_id)
            dropped_for_budget.append(removable)

        selected_chunk_ids = [candidate.chunk.chunk_id for candidate in selected]
        dropped_chunk_ids = [
            candidate.chunk.chunk_id
            for candidate in candidates
            if candidate.chunk.chunk_id not in selected_ids
        ]

        reasoning = self._build_reasoning(
            total_chunks=len(candidates),
            selected_count=len(selected_chunk_ids),
            dropped_count=len(dropped_chunk_ids),
            budget_drop_count=len(dropped_for_budget),
            selected_tokens=self._total_tokens(selected),
            requested_sub_query_count=len(normalized_sub_queries),
        )
        return ContextBudgetDecision(
            max_final_context_tokens=self._max_final_context_tokens,
            max_final_chunks=self._max_final_chunks,
            max_chunks_per_subquery=self._max_chunks_per_subquery,
            selected_chunk_ids=selected_chunk_ids,
            dropped_chunk_ids=dropped_chunk_ids,
            reasoning=reasoning,
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate tokens without external tokenizer dependency."""

        stripped = text.strip()
        if not stripped:
            return 0
        word_count = len(stripped.split())
        char_estimate = max(1, (len(stripped) + 3) // 4)
        return max(word_count, char_estimate)

    def _dedupe_chunks(self, chunks: list[AgenticRetrievedChunk]) -> list[_Candidate]:
        best_by_id: dict[str, AgenticRetrievedChunk] = {}
        for chunk in chunks:
            existing = best_by_id.get(chunk.chunk_id)
            if existing is None or self._candidate_sort_key(chunk) < self._candidate_sort_key(existing):
                best_by_id[chunk.chunk_id] = chunk

        candidates = [
            _Candidate(
                chunk=chunk,
                token_count=self.estimate_tokens(chunk.content),
                sub_queries=self._chunk_sub_queries(chunk),
            )
            for chunk in best_by_id.values()
        ]
        return sorted(candidates, key=lambda candidate: self._candidate_sort_key(candidate.chunk))

    @staticmethod
    def _normalize_sub_queries(sub_queries: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for sub_query in sub_queries:
            value = sub_query.strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        return normalized

    def _group_by_sub_query(
        self,
        candidates: list[_Candidate],
        sub_queries: list[str],
    ) -> dict[str, list[_Candidate]]:
        grouped: dict[str, list[_Candidate]] = defaultdict(list)
        query_keys = {sub_query.casefold(): sub_query for sub_query in sub_queries}

        for candidate in candidates:
            for covered_query in candidate.sub_queries:
                normalized = query_keys.get(covered_query.casefold())
                if normalized:
                    grouped[normalized].append(candidate)

        for sub_query in list(grouped):
            grouped[sub_query] = sorted(
                grouped[sub_query],
                key=lambda candidate: self._coverage_sort_key(candidate, sub_query),
            )
        return grouped

    def _can_add(
        self,
        candidate: _Candidate,
        selected: list[_Candidate],
        selected_ids: set[str],
    ) -> bool:
        if candidate.chunk.chunk_id in selected_ids:
            return False
        if len(selected) >= self._max_final_chunks:
            return False
        for sub_query in candidate.sub_queries:
            if self._selected_count_for_sub_query(selected, sub_query) >= self._max_chunks_per_subquery:
                return False
        return True

    @staticmethod
    def _add_candidate(
        candidate: _Candidate,
        selected: list[_Candidate],
        selected_ids: set[str],
    ) -> None:
        selected.append(candidate)
        selected_ids.add(candidate.chunk.chunk_id)

    def _sort_for_fill(
        self,
        candidates: list[_Candidate],
        selected: list[_Candidate],
    ) -> list[_Candidate]:
        selected_sources = {candidate.chunk.source for candidate in selected}
        return sorted(
            candidates,
            key=lambda candidate: (
                0 if candidate.chunk.source not in selected_sources else 1,
                -candidate.chunk.score,
                self._source_priority(candidate.chunk.source),
                candidate.token_count,
                candidate.chunk.chunk_id,
            ),
        )

    def _choose_budget_drop(
        self,
        selected: list[_Candidate],
        sub_queries: list[str],
    ) -> _Candidate:
        protected_ids = self._coverage_protected_ids(selected, sub_queries)
        removable = [
            candidate
            for candidate in selected
            if candidate.chunk.chunk_id not in protected_ids
        ] or selected
        return sorted(
            removable,
            key=lambda candidate: (
                candidate.chunk.score,
                -candidate.token_count,
                -len(candidate.sub_queries),
                candidate.chunk.chunk_id,
            ),
        )[0]

    def _coverage_protected_ids(
        self,
        selected: list[_Candidate],
        sub_queries: list[str],
    ) -> set[str]:
        protected: set[str] = set()
        for sub_query in sub_queries:
            covering = [
                candidate
                for candidate in selected
                if sub_query.casefold() in {value.casefold() for value in candidate.sub_queries}
            ]
            if len(covering) == 1:
                protected.add(covering[0].chunk.chunk_id)
        return protected

    @staticmethod
    def _chunk_sub_queries(chunk: AgenticRetrievedChunk) -> tuple[str, ...]:
        metadata = chunk.metadata
        raw_values: list[str] = []
        covered = metadata.get("covered_sub_queries")
        if isinstance(covered, list):
            raw_values.extend(value for value in covered if isinstance(value, str))
        single = metadata.get("covered_sub_query")
        if isinstance(single, str):
            raw_values.append(single)

        normalized: list[str] = []
        seen: set[str] = set()
        for value in raw_values:
            stripped = value.strip()
            if not stripped:
                continue
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(stripped)
        return tuple(normalized)

    @staticmethod
    def _source_priority(source: ChunkSource) -> int:
        priorities = {
            ChunkSource.KG: 0,
            ChunkSource.VECTOR: 1,
            ChunkSource.WEB: 2,
        }
        return priorities.get(source, 99)

    def _candidate_sort_key(self, chunk: AgenticRetrievedChunk) -> tuple[float, int, str]:
        return (-chunk.score, self._source_priority(chunk.source), chunk.chunk_id)

    def _coverage_sort_key(self, candidate: _Candidate, sub_query: str) -> tuple[int, float, int, str]:
        exact_match = 0 if sub_query in candidate.sub_queries else 1
        return (
            exact_match,
            -candidate.chunk.score,
            candidate.token_count,
            candidate.chunk.chunk_id,
        )

    @staticmethod
    def _selected_count_for_sub_query(selected: list[_Candidate], sub_query: str) -> int:
        key = sub_query.casefold()
        return sum(
            1
            for candidate in selected
            if key in {value.casefold() for value in candidate.sub_queries}
        )

    @staticmethod
    def _total_tokens(candidates: list[_Candidate]) -> int:
        return sum(candidate.token_count for candidate in candidates)

    @staticmethod
    def _build_reasoning(
        *,
        total_chunks: int,
        selected_count: int,
        dropped_count: int,
        budget_drop_count: int,
        selected_tokens: int,
        requested_sub_query_count: int,
    ) -> str:
        return (
            "selected "
            f"{selected_count}/{total_chunks} chunks for {requested_sub_query_count} sub-queries; "
            f"estimated final context tokens={selected_tokens}; "
            f"dropped={dropped_count}, budget_drops={budget_drop_count}"
        )


__all__ = ["ContextBudgetManager"]
