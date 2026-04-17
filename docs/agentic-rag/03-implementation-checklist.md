# Agentic RAG Implementation Checklist

Version: 1.0
Date: 2026-04-15
Audience: Junior backend developer

## How To Use This Checklist

Work in order. Do not skip phases.

Each phase has:

- Goal
- Files
- Tasks
- Acceptance criteria
- Validation commands

Do not start API integration until component tests for earlier phases pass.

## Phase 0: Baseline

Goal: verify current system behavior before adding agentic mode.

Tasks:

- [ ] Run backend compile check.
- [ ] Start backend and frontend locally.
- [ ] Upload or reuse indexed document.
- [ ] Verify `/api/v1/rag/chat/{workspace_id}/stream` returns answer.
- [ ] Verify sources render.
- [ ] Verify image refs render.
- [ ] Verify `/api/v1/rag/query/{workspace_id}` returns chunks.
- [ ] Record current sample request and response in notes.

Validation:

```powershell
cd backend
uv run python -m compileall app
```

Acceptance:

- Current system still works before any implementation.

## Phase 1: Models, Config, Prompts

Goal: add type-safe foundations.

Files to create:

- `backend/app/services/agentic/__init__.py`
- `backend/app/services/agentic/models.py`
- `backend/app/services/agentic/prompts.py`

Files to edit:

- `backend/app/core/config.py`

Tasks:

- [ ] Create `services/agentic` package.
- [ ] Define enums: `QueryComplexity`, `ChunkSource`, `RewriteStrategy`.
- [ ] Define `QueryAnalysisResult`.
- [ ] Define `ExecutionItem`.
- [ ] Define `ExecutionPlan`.
- [ ] Define `AgenticRetrievedChunk`.
- [ ] Define `RetrievalResult`.
- [ ] Define `SufficiencyJudgment`.
- [ ] Define `RewrittenQuery`.
- [ ] Define `ContextBudgetDecision`.
- [ ] Define `SubQuerySummary`.
- [ ] Define `ResponseJudgment`.
- [ ] Define `ContinuationState`.
- [ ] Define `AgenticRAGState`.
- [ ] Add all config defaults.
- [ ] Add prompt constants for analyzer, planner, sufficiency judge, synthesizer, response judge.

Naming rules:

- [ ] Do not create another model named `RetrievedChunk`.
- [ ] Use clear names ending in `Result`, `Decision`, `Judgment`, or `State`.

Acceptance:

- [ ] All models import without circular dependency.
- [ ] Defaults are safe.
- [ ] `AGENTIC_RAG_ENABLED=false` by default.

Validation:

```powershell
cd backend
uv run python -m compileall app
```

## Phase 2: WebSearchTool

Goal: add optional provider-backed web search, disabled by default.

Files to create:

- `backend/app/services/agentic/web_search_tool.py`

Files to edit:

- `backend/app/core/config.py`
- `backend/pyproject.toml` if adding `openai`

Tasks:

- [ ] Add config: `AGENTIC_WEB_SEARCH_ENABLED`.
- [ ] Add config: `WEB_SEARCH_TOOL_BACKEND`.
- [ ] Add config: `WEB_SEARCH_OPENAI_MODEL`.
- [ ] Add config: `WEB_SEARCH_GOOGLE_MODEL`.
- [ ] Add config: `WEB_SEARCH_TIMEOUT_SECONDS`.
- [ ] Add config: `WEB_SEARCH_MAX_QUERIES`.
- [ ] Add config: `OPENAI_BASE_URL`.
- [ ] Add config: `GOOGLE_BASE_URL`.
- [ ] Implement query normalization.
- [ ] Implement backend resolution.
- [ ] Implement credential checks.
- [ ] Implement OpenAI backend if dependency is available.
- [ ] Implement Google backend.
- [ ] Implement source extraction from OpenAI raw response.
- [ ] Implement source extraction from Google grounding metadata.
- [ ] Implement stable conversion to web evidence chunks.
- [ ] Ensure disabled flag blocks all external calls.

Acceptance:

- [ ] Disabled web search never calls providers.
- [ ] Missing credentials produce controlled error.
- [ ] Duplicate queries are removed.
- [ ] Query count is capped.
- [ ] Output includes provider, model, sources, and search queries.

Validation:

```powershell
cd backend
uv run python -m compileall app
```

## Phase 3: ContextBudgetManager

Goal: prevent final prompt overload.

Files to create:

- `backend/app/services/agentic/context_budget_manager.py`

Tasks:

- [ ] Implement token estimate helper.
- [ ] Group chunks by `covered_sub_query`.
- [ ] Select best chunk per sub-query.
- [ ] Fill remaining slots by score.
- [ ] Enforce `AGENTIC_MAX_FINAL_CHUNKS`.
- [ ] Enforce `AGENTIC_MAX_CHUNKS_PER_SUBQUERY`.
- [ ] Enforce `AGENTIC_MAX_FINAL_CONTEXT_TOKENS`.
- [ ] Return selected and dropped chunk IDs.
- [ ] Add deterministic sort order for ties.

Acceptance:

- [ ] Same input always returns same selected IDs.
- [ ] Every sub-query with evidence keeps at least one chunk when budget allows.
- [ ] Output explains dropped chunks.

Suggested tests:

- [ ] 1 sub-query, 20 chunks.
- [ ] 4 sub-queries, uneven evidence.
- [ ] long chunks exceeding budget.
- [ ] tied scores.

## Phase 4: ParallelRetrievalCoordinator

Goal: run retrieval per sub-query while reusing current RAG services.

Files to create:

- `backend/app/services/agentic/parallel_retrieval.py`

Tasks:

- [ ] Accept db session, workspace ID, sub-queries, entities, document IDs.
- [ ] Use `get_rag_service`.
- [ ] Call `query_deep` when service is `AgenticRAGService`.
- [ ] Fallback to legacy `query` when needed.
- [ ] Run retrieval tasks with timeout.
- [ ] Convert chunks to `AgenticRetrievedChunk`.
- [ ] Preserve citation metadata.
- [ ] Preserve image and table refs.
- [ ] Add `covered_sub_query`.
- [ ] Merge results by chunk ID.
- [ ] Merge `covered_sub_queries` for duplicates.

Acceptance:

- [ ] One failing sub-query does not fail all retrieval.
- [ ] Duplicate chunks merge correctly.
- [ ] Image refs still available for chat response.

## Phase 5: QueryAnalyzer

Goal: classify and decompose query.

Files to create:

- `backend/app/services/agentic/query_analyzer.py`

Tasks:

- [ ] Implement LLM structured JSON call.
- [ ] Validate parsed JSON with `QueryAnalysisResult`.
- [ ] Add rule-based fallback.
- [ ] Detect no-retrieval greetings.
- [ ] Detect single-hop factual questions.
- [ ] Detect multi-hop or multi-topic questions.
- [ ] Detect language.
- [ ] Extract entities best-effort.
- [ ] Extract temporal range best-effort.

Acceptance:

- [ ] Analyzer never raises to orchestrator for normal text.
- [ ] Invalid LLM JSON falls back.
- [ ] Vietnamese query works.
- [ ] English query works.

## Phase 6: ResponsePlanner

Goal: split output work into current batch and continuation batch.

Files to create:

- `backend/app/services/agentic/response_planner.py`

Tasks:

- [ ] Implement LLM structured planning.
- [ ] Add deterministic fallback planner.
- [ ] Estimate output tokens per execution item.
- [ ] Sort by priority.
- [ ] Build `batch_now`.
- [ ] Build `batch_later`.
- [ ] Build continuation message.
- [ ] Ensure every item has stable `item_id`.

Acceptance:

- [ ] Simple question has one item and no continuation.
- [ ] Multi-request question is split when estimated output exceeds config.
- [ ] `batch_now` is never empty unless no retrieval is needed.

## Phase 7: SufficiencyJudge

Goal: decide if retrieved evidence is enough.

Files to create:

- `backend/app/services/agentic/sufficiency_judge.py`

Tasks:

- [ ] Build compact evidence list for judge prompt.
- [ ] Include original query.
- [ ] Include execution plan batch_now.
- [ ] Include analysis.
- [ ] Parse structured JSON.
- [ ] Apply confidence threshold.
- [ ] Detect no chunks as insufficient.
- [ ] Detect overloaded or unfocused context.

Acceptance:

- [ ] Missing data returns missing aspects.
- [ ] Empty chunks returns insufficient.
- [ ] Judge timeout degrades safely.

## Phase 8: QueryRewriter

Goal: improve retrieval when sufficiency fails.

Files to create:

- `backend/app/services/agentic/query_rewriter.py`

Tasks:

- [ ] Implement expansion rewrite.
- [ ] Implement step-back rewrite.
- [ ] Implement hyde rewrite.
- [ ] Include missing aspects in prompt.
- [ ] Track iteration.
- [ ] Return original and rewritten query.

Acceptance:

- [ ] Rewrite output is non-empty.
- [ ] Strategy is preserved.
- [ ] Iteration is preserved.

## Phase 9: HierarchicalSynthesizer

Goal: build compact final context by sub-query.

Files to create:

- `backend/app/services/agentic/hierarchical_synthesizer.py`

Tasks:

- [ ] Group selected chunks by sub-query.
- [ ] Summarize each group with LLM.
- [ ] Preserve supporting chunk IDs.
- [ ] Build assembled context.
- [ ] Include critical evidence section.
- [ ] Fallback to raw selected chunks if summary fails.

Acceptance:

- [ ] Assembled context is shorter than raw selected chunks in normal cases.
- [ ] Every summary lists supporting chunk IDs.
- [ ] No selected chunk loses citation mapping.

## Phase 10: ResponseJudge

Goal: check final answer quality before returning.

Files to create:

- `backend/app/services/agentic/response_judge.py`

Tasks:

- [ ] Judge faithfulness against selected evidence.
- [ ] Judge completeness against original query.
- [ ] Judge coverage against `batch_now`.
- [ ] Parse structured output.
- [ ] Enforce thresholds.
- [ ] Return issues for logs.

Acceptance:

- [ ] Unsupported answer fails.
- [ ] Incomplete batch answer fails.
- [ ] Good grounded answer passes.
- [ ] Timeout degrades safely.

## Phase 11: ContinuationManager

Goal: save and resume unfinished work.

Files to create:

- `backend/app/services/agentic/continuation_manager.py`
- `backend/app/models/agentic_session.py`
- Alembic migration file.

Tasks:

- [ ] Add `AgenticSession` model.
- [ ] Add migration for `agentic_sessions`.
- [ ] Implement save active state.
- [ ] Implement load active session by workspace/session.
- [ ] Implement continuation intent detection.
- [ ] Implement mark completed.
- [ ] Implement expiration.
- [ ] Ensure state JSON stores schema version.

Acceptance:

- [ ] Partial response creates active session.
- [ ] Resume loads remaining item IDs.
- [ ] Completed continuation marks session completed.
- [ ] Expired session is ignored.

## Phase 12: AgenticRAGOrchestrator

Goal: integrate components into one controlled state machine.

Files to create:

- `backend/app/services/agentic/orchestrator.py`

Tasks:

- [ ] Initialize components through constructor injection.
- [ ] Create `AgenticRAGState`.
- [ ] Detect continuation request.
- [ ] Run QueryAnalyzer.
- [ ] Run ResponsePlanner.
- [ ] Map `batch_now` to active sub-queries.
- [ ] Run retrieval attempts loop.
- [ ] Run sufficiency judge per attempt.
- [ ] Run query rewrite when needed.
- [ ] Run web search only when enabled and needed.
- [ ] Run context budget selection.
- [ ] Run hierarchical synthesis.
- [ ] Generate final answer.
- [ ] Run response judge.
- [ ] Save continuation state if needed.
- [ ] Return final state.
- [ ] Provide stream event generator.

Acceptance:

- [ ] Max retrieval attempts enforced.
- [ ] Max replan attempts enforced.
- [ ] Web search not called when disabled.
- [ ] Agentic failure falls back to current hybrid answer path.

## Phase 13: API Integration

Goal: wire orchestrator into existing endpoints without breaking UI.

Files to edit:

- `backend/app/api/rag.py`
- `backend/app/api/chat_agent.py`
- `backend/app/schemas/rag.py`

Tasks:

- [ ] In non-stream chat, branch on `AGENTIC_RAG_ENABLED`.
- [ ] In stream chat, branch on `AGENTIC_RAG_ENABLED`.
- [ ] Preserve current SSE event names.
- [ ] Add optional `agentic_metadata` to complete event.
- [ ] Preserve source format.
- [ ] Preserve image_refs format.
- [ ] Preserve chat message persistence.
- [ ] Persist agent steps as current UI expects.
- [ ] Keep `/rag/query` backward compatible.

Acceptance:

- [ ] Existing FE renders answer.
- [ ] Existing FE renders sources.
- [ ] Existing FE renders images.
- [ ] Extra metadata does not break client.

## Phase 14: Observability

Goal: make agentic behavior debuggable.

Tasks:

- [x] Log run start.
- [x] Log query analysis.
- [x] Log execution plan.
- [x] Log retrieval attempts.
- [x] Log rewrite history.
- [x] Log web search backend and query count.
- [x] Log context budget decision.
- [x] Log response judge result.
- [x] Log continuation state summary.
- [x] Add debug endpoint only if needed and behind existing API style.

Acceptance:

- [x] Logs explain why answer was partial.
- [x] Logs explain why web search was or was not used.
- [x] Logs list selected and dropped chunk IDs.

## Phase 15: Evaluation Dataset

Goal: verify behavior with fixed cases.

Files to create:

- `backend/scripts/agentic_eval_dataset.json`
- `backend/scripts/eval_agentic_rag.py`

Dataset categories:

- [x] 15 no-retrieval questions.
- [x] 20 single-hop questions.
- [x] 20 multi-hop questions.
- [x] 15 multi-request questions.

Each case includes:

- [x] expected complexity
- [x] expected minimum sub-query count
- [x] expected batch split if any
- [x] expected answer coverage notes

Acceptance:

- [x] Evaluation script runs without external web by default.
- [x] Results saved to JSON.

## Final Release Checklist

- [ ] `AGENTIC_RAG_ENABLED=false` path verified.
- [ ] `AGENTIC_RAG_ENABLED=true` internal KB path verified.
- [ ] `AGENTIC_WEB_SEARCH_ENABLED=false` verified.
- [ ] Web search enabled verified with chosen provider.
- [ ] SSE stream verified in UI.
- [ ] Source citations verified.
- [ ] Image refs verified.
- [ ] Continuation verified.
- [ ] Logs verified.
- [ ] Compile check passes.
- [ ] Unit tests pass.
- [ ] No parser/vector/KG schema changes included.

## Recommended PR Order

1. PR-01: models, config, prompts.
2. PR-02: web search tool behind disabled flag.
3. PR-03: context budget manager.
4. PR-04: parallel retrieval coordinator.
5. PR-05: analyzer and planner.
6. PR-06: sufficiency judge and query rewriter.
7. PR-07: hierarchical synthesizer and response judge.
8. PR-08: continuation manager and migration.
9. PR-09: orchestrator.
10. PR-10: API/SSE integration.
11. PR-11: observability and evaluation scripts.

## Manual Verification Script

After integration, manually test:

1. Simple login question.
2. Question asking for screenshot/image evidence.
3. Multi-topic BESTmed question.
4. Broad question requiring several document sections.
5. Query not present in KB with web disabled.
6. Query not present in KB with web enabled.
7. Multi-request query that triggers continuation.
8. "tiếp tục" after continuation offer.
