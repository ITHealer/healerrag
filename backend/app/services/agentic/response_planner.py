"""Output planning for Agentic RAG responses."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.services.agentic.models import ExecutionItem, ExecutionPlan, QueryAnalysisResult, QueryComplexity
from app.services.agentic.prompts import RESPONSE_PLANNER_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """Split user-facing work into current-turn and continuation batches."""

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        max_output_tokens_per_turn: int = 1800,
        timeout_seconds: float = 5.0,
        max_tokens: int = 1500,
        use_llm: bool = True,
    ) -> None:
        self._llm_provider = llm_provider
        self._max_output_tokens_per_turn = max(200, int(max_output_tokens_per_turn))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tokens = max(256, int(max_tokens))
        self._use_llm = use_llm

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        config: Any | None = None,
    ) -> "ResponsePlanner":
        """Build planner from app settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            max_output_tokens_per_turn=config.AGENTIC_MAX_OUTPUT_TOKENS_PER_TURN,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            use_llm=True,
        )

    async def plan(self, *, query: str, analysis: QueryAnalysisResult) -> ExecutionPlan:
        """Create an execution plan, falling back to deterministic planning."""

        if self._use_llm:
            try:
                plan = await asyncio.wait_for(
                    self._plan_with_llm(query=query, analysis=analysis),
                    timeout=self._timeout_seconds,
                )
                return self._normalize_plan(plan=plan, query=query, analysis=analysis)
            except Exception as exc:
                logger.info("ResponsePlanner fallback used: %s", exc)

        return self._fallback_plan(query=query, analysis=analysis)

    async def _plan_with_llm(self, *, query: str, analysis: QueryAnalysisResult) -> ExecutionPlan:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        user_content = (
            "Return only JSON. Do not wrap it in markdown.\n\n"
            f"Max output tokens this turn: {self._max_output_tokens_per_turn}\n\n"
            f"User query:\n{query}\n\n"
            f"Query analysis JSON:\n{analysis.model_dump_json()}"
        )
        result = await provider.acomplete(
            [LLMMessage(role="user", content=user_content)],
            system_prompt=RESPONSE_PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        text = result.content if isinstance(result, LLMResult) else str(result)
        data = self._extract_json_object(text)
        data = self._coerce_plan_data(data)
        return ExecutionPlan.model_validate(data)

    @staticmethod
    def _coerce_plan_data(data: dict[str, Any]) -> dict[str, Any]:
        coerced = dict(data)
        raw_items = coerced.get("items")
        if not isinstance(raw_items, list):
            raw_items = []
        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                item = {"description": str(item)}
            item_id = str(item.get("item_id") or item.get("id") or f"item_{index}")
            if item_id in seen_ids:
                item_id = f"item_{index}"
            seen_ids.add(item_id)
            items.append(
                {
                    "item_id": item_id,
                    "description": str(item.get("description") or item.get("task") or item_id),
                    "priority": item.get("priority") or index,
                    "estimated_output_tokens": item.get("estimated_output_tokens") or item.get("tokens") or 450,
                    "related_sub_queries": item.get("related_sub_queries") or item.get("sub_queries") or [],
                    "depends_on": item.get("depends_on") or [],
                }
            )
        coerced["items"] = items

        def _batch_ids(raw_value: Any) -> list[str]:
            if not isinstance(raw_value, list):
                return []
            ids: list[str] = []
            for item in raw_value:
                if isinstance(item, dict):
                    item_id = item.get("item_id") or item.get("id")
                    if item_id:
                        ids.append(str(item_id))
                elif item is not None:
                    ids.append(str(item))
            return ids

        coerced["batch_now"] = _batch_ids(coerced.get("batch_now"))
        coerced["batch_later"] = _batch_ids(coerced.get("batch_later"))

        # Defensive: remove IDs that the LLM put in batch_now/batch_later but
        # forgot to include in items. This prevents ExecutionPlan Pydantic validator
        # from raising ValueError and forcing a full fallback to rule-based planning.
        valid_item_ids: set[str] = {item["item_id"] for item in items}
        coerced["batch_now"] = [i for i in coerced["batch_now"] if i in valid_item_ids]
        coerced["batch_later"] = [
            i for i in coerced["batch_later"]
            if i in valid_item_ids and i not in set(coerced["batch_now"])
        ]

        if not coerced["batch_now"] and items:
            coerced["batch_now"] = [items[0]["item_id"]]
            coerced["batch_later"] = [item["item_id"] for item in items[1:]]
        coerced["total_items"] = coerced.get("total_items") or len(items)
        return coerced

    def _fallback_plan(self, *, query: str, analysis: QueryAnalysisResult) -> ExecutionPlan:
        descriptions = self._work_descriptions(query=query, analysis=analysis)
        items = [
            ExecutionItem(
                item_id=f"item_{index}",
                description=description,
                priority=index,
                estimated_output_tokens=self._estimate_output_tokens(description),
                related_sub_queries=self._related_sub_queries(description, analysis),
            )
            for index, description in enumerate(descriptions, start=1)
        ]
        return self._build_plan_from_items(
            items=items,
            language=analysis.language,
            reasoning="Rule-based fallback split request into execution items and applied output budget.",
        )

    def _normalize_plan(
        self,
        *,
        plan: ExecutionPlan,
        query: str,
        analysis: QueryAnalysisResult,
    ) -> ExecutionPlan:
        if not plan.items:
            return self._fallback_plan(query=query, analysis=analysis)

        normalized_items: list[ExecutionItem] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(plan.items, start=1):
            item_id = item.item_id.strip() or f"item_{index}"
            if item_id in seen_ids:
                item_id = f"item_{index}"
            seen_ids.add(item_id)
            normalized_items.append(
                item.model_copy(
                    update={
                        "item_id": item_id,
                        "priority": max(1, item.priority),
                        "estimated_output_tokens": max(
                            1,
                            item.estimated_output_tokens or self._estimate_output_tokens(item.description),
                        ),
                    }
                )
            )

        batch_ids = {item.item_id for item in normalized_items}
        batch_now = [item_id for item_id in plan.batch_now if item_id in batch_ids]
        batch_later = [item_id for item_id in plan.batch_later if item_id in batch_ids and item_id not in batch_now]
        batch_now_tokens = sum(
            item.estimated_output_tokens
            for item in normalized_items
            if item.item_id in set(batch_now)
        )
        if not batch_now:
            rebuilt = self._build_plan_from_items(
                items=normalized_items,
                language=analysis.language,
                reasoning="LLM plan items were valid but batches were rebuilt to satisfy budget.",
            )
            return rebuilt
        if batch_now_tokens > self._max_output_tokens_per_turn:
            return self._build_plan_from_items(
                items=normalized_items,
                language=analysis.language,
                reasoning="LLM plan exceeded output budget, so batches were rebuilt.",
            )

        missing = [item.item_id for item in normalized_items if item.item_id not in set(batch_now) | set(batch_later)]
        batch_later.extend(missing)
        can_fully_answer_now = not batch_later
        continuation_message = plan.continuation_message
        if batch_later and not continuation_message:
            continuation_message = self._continuation_message(
                language=analysis.language,
                batch_now=batch_now,
                batch_later=batch_later,
                items=normalized_items,
            )

        return ExecutionPlan(
            can_fully_answer_now=can_fully_answer_now,
            total_items=len(normalized_items),
            items=normalized_items,
            batch_now=batch_now,
            batch_later=batch_later,
            continuation_message=continuation_message,
            reasoning=plan.reasoning or "LLM structured plan normalized by ResponsePlanner.",
        )

    def _build_plan_from_items(
        self,
        *,
        items: list[ExecutionItem],
        language: str,
        reasoning: str,
    ) -> ExecutionPlan:
        if not items:
            items = [
                ExecutionItem(
                    item_id="item_1",
                    description="Respond to the user request.",
                    priority=1,
                    estimated_output_tokens=250,
                    related_sub_queries=[],
                )
            ]

        sorted_items = sorted(items, key=lambda item: (item.priority, item.item_id))
        batch_now: list[str] = []
        batch_later: list[str] = []
        used_tokens = 0
        for item in sorted_items:
            if not batch_now or used_tokens + item.estimated_output_tokens <= self._max_output_tokens_per_turn:
                batch_now.append(item.item_id)
                used_tokens += item.estimated_output_tokens
            else:
                batch_later.append(item.item_id)

        can_fully_answer_now = not batch_later
        continuation_message = None
        if batch_later:
            continuation_message = self._continuation_message(
                language=language,
                batch_now=batch_now,
                batch_later=batch_later,
                items=sorted_items,
            )

        return ExecutionPlan(
            can_fully_answer_now=can_fully_answer_now,
            total_items=len(sorted_items),
            items=sorted_items,
            batch_now=batch_now,
            batch_later=batch_later,
            continuation_message=continuation_message,
            reasoning=reasoning,
        )

    @staticmethod
    def _work_descriptions(*, query: str, analysis: QueryAnalysisResult) -> list[str]:
        if analysis.complexity == QueryComplexity.NO_RETRIEVAL:
            return [query.strip() or "Respond to the user."]
        sub_queries = [item.strip() for item in analysis.sub_queries if item.strip()]
        if len(sub_queries) > 1:
            return sub_queries
        split = ResponsePlanner._split_requested_tasks(query)
        return split or sub_queries or [query.strip()]

    @staticmethod
    def _split_requested_tasks(query: str) -> list[str]:
        parts = [part.strip(" .?") for part in re.split(r"\s*(?:;|\n|\bv\u00e0\b|\band\b|,)\s*", query, flags=re.IGNORECASE)]
        parts = [part for part in parts if len(part.split()) >= 2]
        if 1 < len(parts) <= 8:
            return parts
        return []

    @staticmethod
    def _estimate_output_tokens(description: str) -> int:
        normalized = description.casefold()
        if any(term in normalized for term in ("analyze", "analysis", "ph\u00e2n t\u00edch", "valuation", "\u0111\u1ecbnh gi\u00e1")):
            return 700
        if any(term in normalized for term in ("compare", "comparison", "so s\u00e1nh", "trend", "xu h\u01b0\u1edbng")):
            return 600
        if any(term in normalized for term in ("list", "steps", "c\u00e1c b\u01b0\u1edbc", "how", "c\u00e1ch")):
            return 350
        return 450

    @staticmethod
    def _related_sub_queries(description: str, analysis: QueryAnalysisResult) -> list[str]:
        if not analysis.sub_queries:
            return []
        description_key = description.casefold()
        related = [
            sub_query
            for sub_query in analysis.sub_queries
            if sub_query.casefold() in description_key or description_key in sub_query.casefold()
        ]
        return related or [description]

    @staticmethod
    def _continuation_message(
        *,
        language: str,
        batch_now: list[str],
        batch_later: list[str],
        items: list[ExecutionItem],
    ) -> str:
        item_by_id = {item.item_id: item for item in items}
        now_desc = [item_by_id[item_id].description for item_id in batch_now if item_id in item_by_id]
        later_desc = [item_by_id[item_id].description for item_id in batch_later if item_id in item_by_id]
        if language == "vi":
            return (
                "T\u00f4i s\u1ebd x\u1eed l\u00fd tr\u01b0\u1edbc: "
                + "; ".join(now_desc)
                + ". C\u00f2n l\u1ea1i: "
                + "; ".join(later_desc)
                + ". B\u1ea1n c\u00f3 th\u1ec3 n\u00f3i 'ti\u1ebfp t\u1ee5c' \u0111\u1ec3 x\u1eed l\u00fd ph\u1ea7n c\u00f2n l\u1ea1i."
            )
        return (
            "I will handle first: "
            + "; ".join(now_desc)
            + ". Remaining: "
            + "; ".join(later_desc)
            + ". Say 'continue' to process the remaining items."
        )

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


__all__ = ["ResponsePlanner"]
