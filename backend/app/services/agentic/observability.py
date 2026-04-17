"""Observability helpers for Agentic RAG runs.

The functions in this module intentionally avoid raw query text, chunk content,
and provider credentials. They expose counts, IDs, and decisions that are enough
to debug the pipeline without leaking sensitive document data into logs.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.services.agentic.models import AgenticRAGState


def agentic_log_extra(
    *,
    run_id: str,
    workspace_id: int | str,
    session_id: str | None = None,
    state: AgenticRAGState | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build JSON-safe structured logging metadata for one agentic event."""

    payload: dict[str, Any] = {
        "agentic_run_id": run_id,
        "workspace_id": str(workspace_id),
    }
    if session_id:
        payload["session_id"] = session_id

    if state is not None:
        payload.update(state_observability_metadata(state))

    for key, value in extra.items():
        payload[key] = _safe_value(value)
    return payload


def log_agentic_event(
    logger: logging.Logger,
    message: str,
    *,
    run_id: str,
    workspace_id: int | str,
    session_id: str | None = None,
    state: AgenticRAGState | None = None,
    level: int = logging.INFO,
    **extra: Any,
) -> None:
    """Emit a structured Agentic RAG log event."""

    payload = agentic_log_extra(
        run_id=run_id,
        workspace_id=workspace_id,
        session_id=session_id,
        state=state,
        **extra,
    )
    logger.log(
        level,
        "%s | %s",
        message,
        _format_log_summary(payload),
        extra=payload,
    )


def state_observability_metadata(state: AgenticRAGState) -> dict[str, Any]:
    """Return safe metadata that explains the current Agentic RAG state."""

    analysis = state.analysis
    execution_plan = state.execution_plan
    context_budget = state.context_budget
    response_judgment = state.response_judgment
    sufficiency = state.sufficiency
    continuation = state.continuation_state

    batch_now = execution_plan.batch_now if execution_plan else []
    batch_later = (
        continuation.remaining_item_ids
        if continuation
        else execution_plan.batch_later if execution_plan else []
    )

    return {
        "complexity": analysis.complexity.value if analysis else None,
        "language": analysis.language if analysis else None,
        "sub_query_count": len(analysis.sub_queries) if analysis else 0,
        "entity_count": len(analysis.entities) if analysis else 0,
        "execution_item_count": execution_plan.total_items if execution_plan else 0,
        "batch_now_count": len(batch_now),
        "batch_later_count": len(batch_later),
        "batch_now_item_ids": list(batch_now),
        "batch_later_item_ids": list(batch_later),
        "retrieval_attempts": state.retrieval_attempts,
        "retrieval_result_count": len(state.retrieval_results),
        "merged_chunk_count": len(state.merged_chunks),
        "rewrite_count": len(state.rewrite_history),
        "rewrite_strategies": [item.strategy.value for item in state.rewrite_history],
        "web_search_used": any(source.value == "web" for source in state.sources_used)
        or any(chunk.source.value == "web" for chunk in state.merged_chunks),
        "selected_chunk_count": len(context_budget.selected_chunk_ids) if context_budget else 0,
        "dropped_chunk_count": len(context_budget.dropped_chunk_ids) if context_budget else 0,
        "selected_chunk_ids": list(context_budget.selected_chunk_ids) if context_budget else [],
        "dropped_chunk_ids": list(context_budget.dropped_chunk_ids) if context_budget else [],
        "sufficiency_pass": sufficiency.is_sufficient if sufficiency else None,
        "sufficiency_confidence": sufficiency.confidence if sufficiency else None,
        "missing_aspect_count": len(sufficiency.missing_aspects) if sufficiency else 0,
        "covered_aspect_count": len(sufficiency.covered_aspects) if sufficiency else 0,
        "response_judge_pass": response_judgment.pass_judge if response_judgment else None,
        "faithfulness_score": response_judgment.faithfulness_score if response_judgment else None,
        "completeness_score": response_judgment.completeness_score if response_judgment else None,
        "response_judge_issue_count": len(response_judgment.issues) if response_judgment else 0,
        "continuation_offered": state.continuation_offered,
        "remaining_items_count": len(batch_later),
        "sources_used": [source.value for source in state.sources_used],
    }


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_safe_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return str(value)


def _format_log_summary(payload: dict[str, Any]) -> str:
    keys = [
        "agentic_run_id",
        "workspace_id",
        "session_id",
        "complexity",
        "sub_query_count",
        "execution_item_count",
        "batch_now_count",
        "batch_later_count",
        "retrieval_attempt",
        "retrieval_attempts",
        "active_sub_query_count",
        "merged_chunk_count",
        "rewrite_count",
        "rewrite_strategy",
        "web_search_enabled",
        "web_search_backend",
        "web_search_result_count",
        "web_search_used",
        "selected_chunk_count",
        "dropped_chunk_count",
        "response_judge_pass",
        "continuation_offered",
        "remaining_items_count",
        "error_type",
        "reason",
    ]
    parts: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{key}={_compact_value(value)}")
    return " ".join(parts)


def _compact_value(value: Any) -> str:
    if isinstance(value, list):
        if len(value) > 5:
            return "[" + ",".join(str(item) for item in value[:5]) + f",...(+{len(value) - 5})]"
        return "[" + ",".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_compact_value(item)}" for key, item in list(value.items())[:5]) + "}"
    return str(value)


__all__ = ["agentic_log_extra", "log_agentic_event", "state_observability_metadata"]
