"""Agentic RAG orchestration state machine."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.agentic.context_budget_manager import ContextBudgetManager
from app.services.agentic.continuation_manager import ContinuationManager
from app.services.agentic.hierarchical_synthesizer import HierarchicalSynthesizer
from app.services.agentic.models import (
    AgenticRAGState,
    AgenticRetrievedChunk,
    ChunkSource,
    ExecutionPlan,
    QueryAnalysisResult,
    RewriteStrategy,
    RewrittenQuery,
    ResponseJudgment,
    SufficiencyJudgment,
)
from app.services.agentic.observability import agentic_log_extra, log_agentic_event
from app.services.agentic.parallel_retrieval import ParallelRetrievalCoordinator
from app.services.agentic.query_analyzer import QueryAnalyzer
from app.services.agentic.query_rewriter import QueryRewriter
from app.services.agentic.response_judge import ResponseJudge
from app.services.agentic.response_planner import ResponsePlanner
from app.services.agentic.sufficiency_judge import SufficiencyJudge
from app.services.agentic.web_search_tool import WebSearchTool
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)

AgenticEvent = dict[str, Any]
FallbackRunner = Callable[[str], Awaitable[str]]


class AgenticRAGOrchestrator:
    """Coordinate Agentic RAG components without touching parsing/indexing."""

    def __init__(
        self,
        *,
        query_analyzer: QueryAnalyzer | None = None,
        response_planner: ResponsePlanner | None = None,
        parallel_retriever: ParallelRetrievalCoordinator | None = None,
        sufficiency_judge: SufficiencyJudge | None = None,
        query_rewriter: QueryRewriter | None = None,
        context_budget_manager: ContextBudgetManager | None = None,
        hierarchical_synthesizer: HierarchicalSynthesizer | None = None,
        response_judge: ResponseJudge | None = None,
        continuation_manager: ContinuationManager | None = None,
        web_search_tool: WebSearchTool | None = None,
        llm_provider: Any | None = None,
        max_retrieval_attempts: int = 3,
        max_replan_attempts: int = 2,
        replan_on_response_failure: bool = False,
        web_search_enabled: bool = False,
        fallback_runner: FallbackRunner | None = None,
    ) -> None:
        self.query_analyzer = query_analyzer or QueryAnalyzer.from_settings()
        self.response_planner = response_planner or ResponsePlanner.from_settings()
        self.parallel_retriever = parallel_retriever or ParallelRetrievalCoordinator.from_settings()
        self.sufficiency_judge = sufficiency_judge or SufficiencyJudge.from_settings()
        self.query_rewriter = query_rewriter or QueryRewriter.from_settings()
        self.context_budget_manager = context_budget_manager or ContextBudgetManager.from_settings()
        self.hierarchical_synthesizer = hierarchical_synthesizer or HierarchicalSynthesizer.from_settings()
        self.response_judge = response_judge or ResponseJudge.from_settings()
        self.continuation_manager = continuation_manager or ContinuationManager.from_settings()
        self.web_search_tool = web_search_tool
        self.llm_provider = llm_provider
        self.max_retrieval_attempts = max(1, int(max_retrieval_attempts))
        self.max_replan_attempts = max(0, int(max_replan_attempts))
        self.replan_on_response_failure = replan_on_response_failure
        self.web_search_enabled = web_search_enabled
        self.fallback_runner = fallback_runner

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        fallback_runner: FallbackRunner | None = None,
    ) -> "AgenticRAGOrchestrator":
        """Build orchestrator from configured defaults."""

        web_search_tool = None
        if settings.AGENTIC_WEB_SEARCH_ENABLED:
            web_search_tool = WebSearchTool.from_settings()

        return cls(
            llm_provider=llm_provider,
            web_search_tool=web_search_tool,
            max_retrieval_attempts=settings.AGENTIC_RAG_MAX_RETRIEVAL_ATTEMPTS,
            max_replan_attempts=settings.AGENTIC_RAG_MAX_REPLAN_ATTEMPTS,
            replan_on_response_failure=settings.AGENTIC_RAG_REPLAN_ON_RESPONSE_FAIL,
            web_search_enabled=settings.AGENTIC_WEB_SEARCH_ENABLED,
            fallback_runner=fallback_runner,
        )

    async def run(
        self,
        *,
        query: str,
        workspace_id: int,
        db: AsyncSession,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        document_ids: list[int] | None = None,
        metadata_filter: dict | None = None,
        mode: str = "hybrid",
        include_images: bool = False,
    ) -> AgenticRAGState:
        """Run the full Agentic RAG pipeline and return final state."""

        run_id = str(uuid.uuid4())
        state = AgenticRAGState(
            original_query=query,
            workspace_id=str(workspace_id),
            session_id=session_id,
        )
        log_agentic_event(
            logger,
            "Agentic RAG run started",
            run_id=run_id,
            workspace_id=workspace_id,
            session_id=session_id,
            mode=mode,
            document_filter_count=len(document_ids or []),
            metadata_filter_enabled=bool(metadata_filter),
            stream=False,
        )
        try:
            return await self._run_pipeline(
                state=state,
                run_id=run_id,
                query=query,
                workspace_id=workspace_id,
                db=db,
                session_id=session_id,
                history=history or [],
                document_ids=document_ids,
                metadata_filter=metadata_filter,
                mode=mode,
                include_images=include_images,
                emit=None,
                stream_tokens=False,
            )
        except Exception as exc:
            logger.exception(
                "Agentic RAG run failed",
                extra=agentic_log_extra(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    state=state,
                    error_type=type(exc).__name__,
                ),
            )
            if self.fallback_runner is None:
                raise
            fallback_answer = await self.fallback_runner(query)
            state.final_answer = fallback_answer
            state.generated_answer = fallback_answer
            state.response_judgment = ResponseJudgment(
                pass_judge=False,
                faithfulness_score=0.0,
                completeness_score=0.0,
                issues=[f"Agentic pipeline failed and fallback answer was used: {exc}"],
                reasoning="Fallback runner handled agentic failure.",
            )
            return state

    async def run_stream(
        self,
        *,
        query: str,
        workspace_id: int,
        db: AsyncSession,
        session_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        document_ids: list[int] | None = None,
        metadata_filter: dict | None = None,
        mode: str = "hybrid",
        include_images: bool = False,
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Run the pipeline and yield internal agentic events."""

        run_id = str(uuid.uuid4())
        queue: asyncio.Queue[AgenticEvent | None] = asyncio.Queue()

        async def emit(event: AgenticEvent) -> None:
            await queue.put(event)

        try:
            state = AgenticRAGState(
                original_query=query,
                workspace_id=str(workspace_id),
                session_id=session_id,
            )
            log_agentic_event(
                logger,
                "Agentic RAG stream started",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                mode=mode,
                document_filter_count=len(document_ids or []),
                metadata_filter_enabled=bool(metadata_filter),
                stream=True,
            )
            task = asyncio.create_task(
                self._run_pipeline(
                    state=state,
                    run_id=run_id,
                    query=query,
                    workspace_id=workspace_id,
                    db=db,
                    session_id=session_id,
                    history=history or [],
                    document_ids=document_ids,
                    metadata_filter=metadata_filter,
                    mode=mode,
                    include_images=include_images,
                    emit=emit,
                    stream_tokens=True,
                )
            )
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if event is not None:
                    yield event
            state = await task
            yield {"event": "run_complete", "data": {"state": state}}
        except Exception as exc:
            logger.exception(
                "Agentic RAG stream failed",
                extra=agentic_log_extra(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    error_type=type(exc).__name__,
                ),
            )
            if self.fallback_runner is None:
                yield {"event": "run_error", "data": {"message": str(exc)}}
                return
            fallback_answer = await self.fallback_runner(query)
            yield {"event": "generation_token", "data": {"text": fallback_answer}}
            fallback_state = AgenticRAGState(
                original_query=query,
                workspace_id=str(workspace_id),
                session_id=session_id,
                final_answer=fallback_answer,
                generated_answer=fallback_answer,
            )
            yield {"event": "run_complete", "data": {"state": fallback_state}}

    async def _run_pipeline(
        self,
        *,
        state: AgenticRAGState,
        run_id: str,
        query: str,
        workspace_id: int,
        db: AsyncSession,
        session_id: str | None,
        history: list[dict[str, Any]],
        document_ids: list[int] | None,
        metadata_filter: dict | None,
        mode: str,
        include_images: bool,
        emit: Callable[[AgenticEvent], Awaitable[None]] | None,
        stream_tokens: bool,
    ) -> AgenticRAGState:
        await self._emit(emit, "analysis_started", {"step": "analyzing", "detail": "Analyzing your question..."})
        continuation_state = None
        is_continuation = bool(session_id and self.continuation_manager.is_continuation_intent(query))
        if is_continuation and session_id:
            continuation_state = await self.continuation_manager.load_active(
                db=db,
                workspace_id=workspace_id,
                session_id=session_id,
            )

        if continuation_state:
            state.continuation_state = continuation_state
            state.original_query = continuation_state.original_query
            state.execution_plan = self._plan_for_continuation(continuation_state.execution_plan)
            state.analysis = await self.query_analyzer.analyze(continuation_state.original_query, history)
            log_agentic_event(
                logger,
                "Agentic continuation state loaded",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                continuation_remaining_count=len(continuation_state.remaining_item_ids),
            )
        else:
            # [1] Query Analysis — classify complexity, extract entities, sub-queries
            if settings.AGENTIC_ENABLE_QUERY_ANALYSIS:
                state.analysis = await self.query_analyzer.analyze(query, history)
            else:
                # Fallback: treat as single_hop, use raw query as-is
                from app.services.agentic.models import QueryComplexity
                state.analysis = QueryAnalysisResult(
                    complexity=QueryComplexity.SINGLE_HOP,
                    sub_queries=[query],
                    entities=[],
                    temporal_range=None,
                    language="auto",
                    strategy_hint="direct",
                    reasoning="Query analysis disabled — treating as single_hop.",
                )
                logger.info("AGENTIC_ENABLE_QUERY_ANALYSIS=false: using single_hop fallback")

            # [2] Response Planning — split into batch_now / batch_later
            if settings.AGENTIC_ENABLE_RESPONSE_PLANNER:
                state.execution_plan = await self.response_planner.plan(query=query, analysis=state.analysis)
            else:
                # Fallback: single execution item covering the whole query
                from app.services.agentic.models import ExecutionItem
                state.execution_plan = ExecutionPlan(
                    items=[ExecutionItem(item_id="item_1", description=query, sub_queries=[query])],
                    batch_now=["item_1"],
                    batch_later=[],
                    continuation_message=None,
                    total_items=1,
                )
                logger.info("AGENTIC_ENABLE_RESPONSE_PLANNER=false: using single-item plan fallback")

        log_agentic_event(
            logger,
            "Agentic query analysis completed",
            run_id=run_id,
            workspace_id=workspace_id,
            session_id=session_id,
            state=state,
            strategy_hint=state.analysis.strategy_hint if state.analysis else None,
            temporal_range=state.analysis.temporal_range if state.analysis else None,
        )
        log_agentic_event(
            logger,
            "Agentic execution plan completed",
            run_id=run_id,
            workspace_id=workspace_id,
            session_id=session_id,
            state=state,
            can_fully_answer_now=state.execution_plan.can_fully_answer_now if state.execution_plan else None,
        )

        await self._emit(
            emit,
            "analysis_done",
            {
                "complexity": state.analysis.complexity.value,
                "sub_query_count": len(state.analysis.sub_queries),
            },
        )
        await self._emit(
            emit,
            "planning_done",
            {
                "total_items": state.execution_plan.total_items if state.execution_plan else 0,
                "batch_now": state.execution_plan.batch_now if state.execution_plan else [],
                "batch_later": state.execution_plan.batch_later if state.execution_plan else [],
            },
        )

        active_sub_queries = self._active_sub_queries(state.analysis, state.execution_plan)
        selected_chunks: list[AgenticRetrievedChunk] = []

        for replan_attempt in range(self.max_replan_attempts + 1):
            state.replan_attempts = replan_attempt
            current_sub_queries = active_sub_queries

            for retrieval_attempt in range(1, self.max_retrieval_attempts + 1):
                state.retrieval_attempts = retrieval_attempt
                await self._emit(
                    emit,
                    "retrieval_started" if retrieval_attempt == 1 else "retrieval_retry",
                    {
                        "attempt": retrieval_attempt,
                        "sub_queries": current_sub_queries,
                    },
                )
                log_agentic_event(
                    logger,
                    "Agentic retrieval attempt started",
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    state=state,
                    retrieval_attempt=retrieval_attempt,
                    replan_attempt=replan_attempt,
                    active_sub_query_count=len(current_sub_queries),
                )
                results = await self.parallel_retriever.retrieve_all(
                    db=db,
                    workspace_id=workspace_id,
                    sub_queries=current_sub_queries,
                    entities=state.analysis.entities if state.analysis else None,
                    document_ids=document_ids,
                    metadata_filter=metadata_filter,
                    mode=mode,
                    include_images=include_images,
                )
                state.retrieval_results = results
                state.merged_chunks = self.parallel_retriever.merge_results(results)
                if results and not state.merged_chunks and all(result.error_message for result in results):
                    state.sufficiency = SufficiencyJudgment(
                        is_sufficient=False,
                        confidence=1.0,
                        missing_aspects=["All retrieval sub-queries failed due to system errors."],
                        covered_aspects=[],
                        suggested_rewrite_strategy=None,
                        reasoning="Retrieval failed before evidence could be judged; retrying the same system failure was skipped.",
                    )
                    log_agentic_event(
                        logger,
                        "Agentic retrieval stopped after system errors",
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        state=state,
                        retrieval_attempt=retrieval_attempt,
                        error_count=len(results),
                    )
                    break
                log_agentic_event(
                    logger,
                    "Agentic retrieval attempt completed",
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    state=state,
                    retrieval_attempt=retrieval_attempt,
                    per_sub_query_chunk_counts=[len(result.chunks) for result in results],
                )
                # [3] Sufficiency Judgment — decide if retrieved evidence is enough
                if not settings.AGENTIC_ENABLE_SUFFICIENCY_JUDGE:
                    # Skip judgment: always treat as sufficient after first retrieval
                    state.sufficiency = SufficiencyJudgment(
                        is_sufficient=True,
                        confidence=1.0,
                        missing_aspects=[],
                        covered_aspects=[],
                        suggested_rewrite_strategy=None,
                        reasoning="Sufficiency judge disabled — assuming sufficient after retrieval.",
                    )
                    logger.info("AGENTIC_ENABLE_SUFFICIENCY_JUDGE=false: skipping judgment, exiting loop")
                    break

                state.sufficiency = await self.sufficiency_judge.judge(
                    original_query=state.original_query,
                    analysis=state.analysis,
                    execution_plan=state.execution_plan,
                    chunks=state.merged_chunks,
                )
                if state.sufficiency.is_sufficient:
                    log_agentic_event(
                        logger,
                        "Agentic sufficiency judge passed",
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        state=state,
                        retrieval_attempt=retrieval_attempt,
                    )
                    break

                if retrieval_attempt >= self.max_retrieval_attempts:
                    log_agentic_event(
                        logger,
                        "Agentic retrieval attempts exhausted",
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        state=state,
                        retrieval_attempt=retrieval_attempt,
                    )
                    await self._maybe_web_search(
                        state=state,
                        query=state.original_query,
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        emit=emit,
                    )
                    break

                strategy = self._rewrite_strategy(state.sufficiency.suggested_rewrite_strategy)
                if strategy == "websearch":
                    log_agentic_event(
                        logger,
                        "Agentic sufficiency requested web search",
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        state=state,
                        retrieval_attempt=retrieval_attempt,
                    )
                    await self._maybe_web_search(
                        state=state,
                        query=state.original_query,
                        run_id=run_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        emit=emit,
                    )
                    break
                # [4] Query Rewriting — rewrite query to improve retrieval on next attempt
                if not settings.AGENTIC_ENABLE_QUERY_REWRITER:
                    logger.info("AGENTIC_ENABLE_QUERY_REWRITER=false: skipping rewrite, exiting retry loop")
                    break

                rewritten = await self.query_rewriter.rewrite(
                    original_query=state.original_query,
                    missing_aspects=state.sufficiency.missing_aspects,
                    strategy=strategy,
                    iteration=retrieval_attempt,
                )
                state.rewrite_history.append(rewritten)
                log_agentic_event(
                    logger,
                    "Agentic query rewrite completed",
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    state=state,
                    retrieval_attempt=retrieval_attempt,
                    rewrite_strategy=rewritten.strategy.value,
                    rewrite_iteration=rewritten.iteration,
                )
                current_sub_queries = [rewritten.rewritten_query]

            state.context_budget = self.context_budget_manager.select(
                chunks=state.merged_chunks,
                sub_queries=current_sub_queries,
            )
            log_agentic_event(
                logger,
                "Agentic context budget selected evidence",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                max_final_context_tokens=state.context_budget.max_final_context_tokens,
                max_final_chunks=state.context_budget.max_final_chunks,
                max_chunks_per_subquery=state.context_budget.max_chunks_per_subquery,
            )
            selected_chunks = [
                chunk for chunk in state.merged_chunks if chunk.chunk_id in state.context_budget.selected_chunk_ids
            ]
            await self._emit(
                emit,
                "sources_selected",
                {
                    "selected_chunk_ids": state.context_budget.selected_chunk_ids,
                    "dropped_chunk_ids": state.context_budget.dropped_chunk_ids,
                    "chunks": selected_chunks,
                },
            )
            await self._emit(emit, "images_selected", {"image_refs": self._image_refs_from_chunks(selected_chunks)})

            # [5] Hierarchical Synthesis — LLM summarize per sub-query, then assemble context
            if settings.AGENTIC_ENABLE_HIERARCHICAL_SYNTHESIS:
                state.subquery_summaries = await self.hierarchical_synthesizer.summarize(
                    chunks=selected_chunks,
                    sub_queries=current_sub_queries,
                )
                state.assembled_context = self.hierarchical_synthesizer.assemble(
                    original_query=state.original_query,
                    summaries=state.subquery_summaries,
                    chunks=selected_chunks,
                )
            else:
                # Skip LLM summarize: build context directly from raw chunk excerpts
                logger.info("AGENTIC_ENABLE_HIERARCHICAL_SYNTHESIS=false: using raw chunk context")
                state.subquery_summaries = []
                raw_parts = [f"[Chunk {i+1}] {c.content[:600]}" for i, c in enumerate(selected_chunks[:8])]
                state.assembled_context = f"Original question: {state.original_query}\n\nEvidence:\n" + "\n\n".join(raw_parts)

            state.generated_answer = await self._generate_answer(
                query=state.original_query,
                context=state.assembled_context,
                emit=emit,
                stream_tokens=stream_tokens,
            )
            # [6] Response Judge — evaluate faithfulness and completeness
            if settings.AGENTIC_ENABLE_RESPONSE_JUDGE:
                state.response_judgment = await self.response_judge.judge(
                    original_query=state.original_query,
                    generated_answer=state.generated_answer,
                    chunks=selected_chunks,
                    execution_plan=state.execution_plan,
                )
            else:
                logger.info("AGENTIC_ENABLE_RESPONSE_JUDGE=false: skipping response judge")
                state.response_judgment = ResponseJudgment(
                    pass_judge=True,
                    faithfulness_score=1.0,
                    completeness_score=1.0,
                    issues=[],
                    reasoning="Response judge disabled.",
                )
            log_agentic_event(
                logger,
                "Agentic response judge completed",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                replan_attempt=replan_attempt,
            )
            await self._emit(
                emit,
                "response_judged",
                {
                    "pass_judge": state.response_judgment.pass_judge,
                    "issues": state.response_judgment.issues,
                },
            )
            if (
                state.response_judgment.pass_judge
                or replan_attempt >= self.max_replan_attempts
                or not self.replan_on_response_failure
                or not selected_chunks
            ):
                break

        state.final_answer = state.generated_answer
        state.citations = self._extract_citations(selected_chunks)
        state.sources_used = sorted({chunk.source for chunk in selected_chunks}, key=lambda item: item.value)

        await self._handle_continuation(
            state=state,
            db=db,
            workspace_id=workspace_id,
            session_id=session_id,
            selected_chunks=selected_chunks,
            run_id=run_id,
        )
        log_agentic_event(
            logger,
            "Agentic RAG run completed",
            run_id=run_id,
            workspace_id=workspace_id,
            session_id=session_id,
            state=state,
        )
        return state

    async def _generate_answer(
        self,
        *,
        query: str,
        context: str,
        emit: Callable[[AgenticEvent], Awaitable[None]] | None,
        stream_tokens: bool,
    ) -> str:
        provider = self.llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self.llm_provider = provider

        from app.services.agentic.prompts import RAG_GENERATION_SYSTEM_PROMPT

        user_message = f"Question: {query}\n\nContext:\n{context}"
        messages = [LLMMessage(role="user", content=user_message)]
        if stream_tokens and hasattr(provider, "astream"):
            answer_parts: list[str] = []
            await self._emit(emit, "generation_started", {"step": "generating", "detail": "Generating answer..."})
            async for chunk in provider.astream(
                messages,
                system_prompt=RAG_GENERATION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            ):
                if getattr(chunk, "type", "") == "text":
                    answer_parts.append(chunk.text)
                    await self._emit(emit, "generation_token", {"text": chunk.text})
            return "".join(answer_parts).strip() or "Unable to generate a response."

        result = await provider.acomplete(
            messages,
            system_prompt=RAG_GENERATION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )
        answer = result.content if isinstance(result, LLMResult) else str(result)
        await self._emit(emit, "generation_token", {"text": answer})
        return answer.strip() or "Unable to generate a response."

    async def _maybe_web_search(
        self,
        *,
        state: AgenticRAGState,
        query: str,
        run_id: str,
        workspace_id: int,
        session_id: str | None,
        emit: Callable[[AgenticEvent], Awaitable[None]] | None,
    ) -> None:
        if not self.web_search_enabled or self.web_search_tool is None:
            log_agentic_event(
                logger,
                "Agentic web search skipped",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                web_search_enabled=self.web_search_enabled,
                web_search_tool_configured=self.web_search_tool is not None,
            )
            return
        try:
            await self._emit(emit, "web_search_started", {"query": query})
            backend = getattr(self.web_search_tool, "_backend", "unknown")
            log_agentic_event(
                logger,
                "Agentic web search started",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                web_search_backend=backend,
            )
            output = await self.web_search_tool.search(query=query)
            state.merged_chunks.extend(self.web_search_tool.to_chunks(output))
            log_agentic_event(
                logger,
                "Agentic web search completed",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                web_search_backend=output.backend,
                web_search_query_count=sum(len(result.search_queries) or 1 for result in output.results),
                web_search_result_count=len(output.results),
            )
        except Exception as exc:
            logger.warning(
                "Agentic web search failed: %s",
                exc,
                extra=agentic_log_extra(
                    run_id=run_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    state=state,
                    error_type=type(exc).__name__,
                ),
            )

    async def _handle_continuation(
        self,
        *,
        state: AgenticRAGState,
        db: AsyncSession,
        workspace_id: int,
        session_id: str | None,
        selected_chunks: list[AgenticRetrievedChunk],
        run_id: str,
    ) -> None:
        if not session_id or not state.execution_plan:
            log_agentic_event(
                logger,
                "Agentic continuation skipped",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                reason="missing_session_or_plan",
            )
            return
        if state.execution_plan.batch_later:
            state.continuation_state = await self.continuation_manager.save(
                db=db,
                workspace_id=workspace_id,
                session_id=session_id,
                original_query=state.original_query,
                execution_plan=state.execution_plan,
                completed_item_ids=state.execution_plan.batch_now,
                remaining_item_ids=state.execution_plan.batch_later,
                evidence_chunk_ids=[chunk.chunk_id for chunk in selected_chunks],
                citations=state.citations,
            )
            if state.continuation_state:
                state.continuation_offered = True
                if state.execution_plan.continuation_message:
                    state.final_answer = f"{state.final_answer}\n\n{state.execution_plan.continuation_message}"
            log_agentic_event(
                logger,
                "Agentic continuation state saved",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
                evidence_chunk_count=len(selected_chunks),
            )
        elif state.continuation_state:
            await self.continuation_manager.mark_completed(
                db=db,
                workspace_id=workspace_id,
                session_id=session_id,
            )
            log_agentic_event(
                logger,
                "Agentic continuation state completed",
                run_id=run_id,
                workspace_id=workspace_id,
                session_id=session_id,
                state=state,
            )

    @staticmethod
    async def _emit(
        emit: Callable[[AgenticEvent], Awaitable[None]] | None,
        event: str,
        data: dict[str, Any],
    ) -> None:
        if emit is not None:
            await emit({"event": event, "data": data})

    @staticmethod
    def _active_sub_queries(
        analysis: QueryAnalysisResult | None,
        execution_plan: ExecutionPlan | None,
    ) -> list[str]:
        if analysis is None:
            return []
        if execution_plan is None or not execution_plan.batch_now:
            return analysis.sub_queries
        item_by_id = {item.item_id: item for item in execution_plan.items}
        sub_queries: list[str] = []
        for item_id in execution_plan.batch_now:
            item = item_by_id.get(item_id)
            if item and item.related_sub_queries:
                sub_queries.extend(item.related_sub_queries)
            elif item:
                sub_queries.append(item.description)
        return sub_queries or analysis.sub_queries

    @staticmethod
    def _plan_for_continuation(plan: ExecutionPlan) -> ExecutionPlan:
        remaining = list(plan.batch_later)
        return plan.model_copy(
            update={
                "can_fully_answer_now": True,
                "batch_now": remaining,
                "batch_later": [],
                "continuation_message": None,
            }
        )

    @staticmethod
    def _rewrite_strategy(raw_strategy: RewriteStrategy | str | None) -> RewriteStrategy | str:
        if isinstance(raw_strategy, RewriteStrategy):
            return raw_strategy
        if isinstance(raw_strategy, str):
            value = raw_strategy.strip().lower()
            if value == "websearch":
                return "websearch"
            try:
                return RewriteStrategy(value)
            except ValueError:
                return RewriteStrategy.EXPANSION
        return RewriteStrategy.EXPANSION

    @staticmethod
    def _extract_citations(chunks: list[AgenticRetrievedChunk]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for chunk in chunks:
            citation = chunk.metadata.get("citation")
            if isinstance(citation, dict):
                citations.append(citation)
            else:
                citations.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_type": chunk.source.value,
                        "document_id": chunk.metadata.get("document_id", 0),
                        "page_no": chunk.metadata.get("page_no", 0),
                    }
                )
        return citations

    @staticmethod
    def _image_refs_from_chunks(chunks: list[AgenticRetrievedChunk]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            for image in chunk.metadata.get("result_image_refs", []) or []:
                image_id = str(image.get("image_id", ""))
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                images.append(image)
        return images


__all__ = ["AgenticRAGOrchestrator", "AgenticEvent"]
