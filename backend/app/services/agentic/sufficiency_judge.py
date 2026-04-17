"""Evidence sufficiency judgment for Agentic RAG retrieval loops."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.services.agentic.models import (
    AgenticRetrievedChunk,
    ExecutionPlan,
    QueryAnalysisResult,
    RewriteStrategy,
    SufficiencyJudgment,
)
from app.services.agentic.prompts import SUFFICIENCY_JUDGE_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class SufficiencyJudge:
    """Decide whether retrieved evidence can support the current answer batch."""

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        sufficiency_threshold: float = 0.7,
        timeout_seconds: float = 5.0,
        max_tokens: int = 1200,
        max_evidence_items: int = 12,
        overload_chunk_count: int = 20,
        use_llm: bool = True,
    ) -> None:
        self._llm_provider = llm_provider
        self._sufficiency_threshold = min(1.0, max(0.0, float(sufficiency_threshold)))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tokens = max(256, int(max_tokens))
        self._max_evidence_items = max(1, int(max_evidence_items))
        self._overload_chunk_count = max(self._max_evidence_items + 1, int(overload_chunk_count))
        self._use_llm = use_llm

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        config: Any | None = None,
    ) -> "SufficiencyJudge":
        """Build judge from app settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            sufficiency_threshold=config.AGENTIC_RAG_SUFFICIENCY_THRESHOLD,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            max_evidence_items=config.AGENTIC_MAX_FINAL_CHUNKS * 2,
            overload_chunk_count=config.AGENTIC_MAX_FINAL_CHUNKS * 3,
            use_llm=True,
        )

    async def judge(
        self,
        *,
        original_query: str,
        analysis: QueryAnalysisResult,
        execution_plan: ExecutionPlan,
        chunks: list[AgenticRetrievedChunk],
    ) -> SufficiencyJudgment:
        """Return a structured sufficiency decision for current evidence."""

        if not chunks:
            return SufficiencyJudgment(
                is_sufficient=False,
                confidence=1.0,
                missing_aspects=["No retrieved evidence was found."],
                covered_aspects=[],
                suggested_rewrite_strategy=RewriteStrategy.EXPANSION,
                reasoning="No chunks were available for the sufficiency judge.",
            )

        if self._use_llm:
            try:
                judgment = await asyncio.wait_for(
                    self._judge_with_llm(
                        original_query=original_query,
                        analysis=analysis,
                        execution_plan=execution_plan,
                        chunks=chunks,
                    ),
                    timeout=self._timeout_seconds,
                )
                return self._normalize_judgment(judgment=judgment, chunks=chunks)
            except Exception as exc:
                logger.info("SufficiencyJudge fallback used: %s", exc)
                return SufficiencyJudgment(
                    is_sufficient=True,
                    confidence=0.35,
                    missing_aspects=[],
                    covered_aspects=["Retrieved evidence exists, but automated judging failed."],
                    suggested_rewrite_strategy=None,
                    reasoning="Judge failed or timed out; degraded safely without blocking retrieval.",
                )

        return self._rule_based_judgment(analysis=analysis, execution_plan=execution_plan, chunks=chunks)

    async def _judge_with_llm(
        self,
        *,
        original_query: str,
        analysis: QueryAnalysisResult,
        execution_plan: ExecutionPlan,
        chunks: list[AgenticRetrievedChunk],
    ) -> SufficiencyJudgment:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        user_content = (
            "Return only JSON. Do not wrap it in markdown.\n\n"
            f"Sufficiency threshold: {self._sufficiency_threshold}\n\n"
            f"Original query:\n{original_query}\n\n"
            f"Query analysis JSON:\n{analysis.model_dump_json()}\n\n"
            f"Execution plan JSON:\n{execution_plan.model_dump_json()}\n\n"
            f"Evidence:\n{self._evidence_preview(chunks)}"
        )
        result = await provider.acomplete(
            [LLMMessage(role="user", content=user_content)],
            system_prompt=SUFFICIENCY_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        text = result.content if isinstance(result, LLMResult) else str(result)
        data = self._extract_json_object(text)
        return SufficiencyJudgment.model_validate(data)

    def _normalize_judgment(
        self,
        *,
        judgment: SufficiencyJudgment,
        chunks: list[AgenticRetrievedChunk],
    ) -> SufficiencyJudgment:
        missing_aspects = list(judgment.missing_aspects)
        covered_aspects = list(judgment.covered_aspects)
        suggested_strategy = judgment.suggested_rewrite_strategy
        is_sufficient = judgment.is_sufficient and judgment.confidence >= self._sufficiency_threshold

        if judgment.is_sufficient and judgment.confidence < self._sufficiency_threshold:
            missing_aspects.append("Judge confidence is below the configured sufficiency threshold.")
            suggested_strategy = suggested_strategy or RewriteStrategy.EXPANSION

        if self._is_overloaded(chunks):
            is_sufficient = False
            missing_aspects.append("Context is overloaded or not focused enough.")
            suggested_strategy = RewriteStrategy.STEP_BACK

        return judgment.model_copy(
            update={
                "is_sufficient": is_sufficient,
                "missing_aspects": self._dedupe(missing_aspects),
                "covered_aspects": self._dedupe(covered_aspects),
                "suggested_rewrite_strategy": suggested_strategy,
            }
        )

    def _rule_based_judgment(
        self,
        *,
        analysis: QueryAnalysisResult,
        execution_plan: ExecutionPlan,
        chunks: list[AgenticRetrievedChunk],
    ) -> SufficiencyJudgment:
        if self._is_overloaded(chunks):
            return SufficiencyJudgment(
                is_sufficient=False,
                confidence=0.45,
                missing_aspects=["Context is overloaded or not focused enough."],
                covered_aspects=[],
                suggested_rewrite_strategy=RewriteStrategy.STEP_BACK,
                reasoning="Rule-based judge detected too many evidence chunks for a focused answer.",
            )

        expected_aspects = self._expected_aspects(analysis=analysis, execution_plan=execution_plan)
        evidence_text = "\n".join(chunk.content for chunk in chunks).casefold()
        covered: list[str] = []
        missing: list[str] = []

        for aspect in expected_aspects:
            if self._aspect_is_covered(aspect, evidence_text):
                covered.append(aspect)
            else:
                missing.append(aspect)

        if not expected_aspects:
            covered = ["Retrieved evidence is available."]

        coverage_ratio = len(covered) / max(1, len(expected_aspects))
        confidence = min(0.95, max(0.4, coverage_ratio))
        is_sufficient = not missing and confidence >= self._sufficiency_threshold
        strategy: RewriteStrategy | None = None
        if missing:
            strategy = RewriteStrategy.EXPANSION

        return SufficiencyJudgment(
            is_sufficient=is_sufficient,
            confidence=confidence,
            missing_aspects=missing,
            covered_aspects=covered,
            suggested_rewrite_strategy=strategy,
            reasoning="Rule-based judge matched requested aspects against retrieved evidence.",
        )

    def _evidence_preview(self, chunks: list[AgenticRetrievedChunk]) -> str:
        preview_items: list[str] = []
        sorted_chunks = sorted(chunks, key=lambda chunk: (-chunk.score, chunk.chunk_id))
        for index, chunk in enumerate(sorted_chunks[: self._max_evidence_items], start=1):
            metadata = {
                "source": chunk.source.value,
                "score": chunk.score,
                "covered_sub_queries": chunk.metadata.get("covered_sub_queries")
                or chunk.metadata.get("covered_sub_query"),
                "covered_entities": chunk.metadata.get("covered_entities"),
            }
            content = re.sub(r"\s+", " ", chunk.content).strip()[:900]
            preview_items.append(
                f"[{index}] chunk_id={chunk.chunk_id}\n"
                f"metadata={json.dumps(metadata, ensure_ascii=True)}\n"
                f"content={content}"
            )
        if len(chunks) > self._max_evidence_items:
            preview_items.append(f"... {len(chunks) - self._max_evidence_items} additional chunks omitted")
        return "\n\n".join(preview_items)

    def _is_overloaded(self, chunks: list[AgenticRetrievedChunk]) -> bool:
        if len(chunks) <= self._overload_chunk_count:
            return False

        covered_sub_queries: set[str] = set()
        for chunk in chunks:
            raw_values = chunk.metadata.get("covered_sub_queries") or chunk.metadata.get("covered_sub_query")
            if isinstance(raw_values, str):
                covered_sub_queries.add(raw_values.casefold())
            elif isinstance(raw_values, list):
                covered_sub_queries.update(str(value).casefold() for value in raw_values if str(value).strip())

        if not covered_sub_queries:
            return True
        average_chunks_per_subquery = len(chunks) / max(1, len(covered_sub_queries))
        return average_chunks_per_subquery > self._max_evidence_items

    @staticmethod
    def _expected_aspects(
        *,
        analysis: QueryAnalysisResult,
        execution_plan: ExecutionPlan,
    ) -> list[str]:
        item_by_id = {item.item_id: item for item in execution_plan.items}
        aspects = [
            item_by_id[item_id].description
            for item_id in execution_plan.batch_now
            if item_id in item_by_id and item_by_id[item_id].description.strip()
        ]
        if not aspects:
            aspects = [sub_query for sub_query in analysis.sub_queries if sub_query.strip()]
        return SufficiencyJudge._dedupe(aspects)

    @staticmethod
    def _aspect_is_covered(aspect: str, evidence_text: str) -> bool:
        tokens = [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9_.$-]{3,}", aspect)
            if token.casefold() not in {"the", "and", "for", "with", "about", "how", "what"}
        ]
        if not tokens:
            return True
        required_matches = 1 if len(tokens) == 1 else max(2, int(len(set(tokens)) * 0.75 + 0.999))
        matches = sum(1 for token in set(tokens) if token in evidence_text)
        return matches >= required_matches

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


__all__ = ["SufficiencyJudge"]
