# Agentic RAG PR Breakdown

Version: 1.0
Date: 2026-04-15

## Purpose

This file converts the implementation phases into reviewable pull requests. Each PR should be small enough for a junior developer to implement and a reviewer to validate without reading the whole future system at once.

## General PR Rules

- Keep each PR focused on one boundary.
- Do not mix API integration with low-level component creation.
- Do not enable `AGENTIC_RAG_ENABLED` by default.
- Add tests with every new component where possible.
- Prefer dependency injection over global imports inside component constructors.
- Keep API files thin.
- Keep existing Hybrid RAG path untouched unless the PR explicitly integrates agentic mode.

## PR-01: Agentic Foundation

Goal: add package, models, config, and prompts.

Depends on: none.

Files:

- `backend/app/services/agentic/__init__.py`
- `backend/app/services/agentic/models.py`
- `backend/app/services/agentic/prompts.py`
- `backend/app/core/config.py`

Tasks:

- [ ] Add agentic package.
- [ ] Add all Pydantic models and enums.
- [ ] Add safe config defaults.
- [ ] Add prompt constants only, no orchestration.
- [ ] Add model validation tests.

Review gates:

- [ ] No runtime behavior changes.
- [ ] No API changes.
- [ ] Config defaults are disabled.
- [ ] Naming is unambiguous.

## PR-02: Provider-backed Web Search Tool

Goal: add web search tool behind `AGENTIC_WEB_SEARCH_ENABLED=false`.

Depends on: PR-01.

Files:

- `backend/app/services/agentic/web_search_tool.py`
- `backend/app/core/config.py`
- `backend/pyproject.toml` if OpenAI SDK is added.

Tasks:

- [ ] Implement backend resolution.
- [ ] Implement Google backend.
- [ ] Implement OpenAI backend if SDK is added.
- [ ] Extract source URLs and titles.
- [ ] Convert results into web evidence format.
- [ ] Add fixture-based extraction tests.

Review gates:

- [ ] Disabled flag prevents provider calls.
- [ ] Missing credentials do not crash import.
- [ ] Web results are never labeled as document chunks.

## PR-03: Context Budget Manager

Goal: add deterministic final-context selection.

Depends on: PR-01.

Files:

- `backend/app/services/agentic/context_budget_manager.py`

Tasks:

- [ ] Estimate tokens.
- [ ] Select best chunks by sub-query.
- [ ] Enforce chunk and token caps.
- [ ] Track selected and dropped IDs.
- [ ] Add deterministic unit tests.

Review gates:

- [ ] No LLM calls in this component.
- [ ] Same input returns same output.
- [ ] Per-sub-query coverage is preserved when possible.

## PR-04: Parallel Retrieval Coordinator

Goal: wrap current retrieval into sub-query parallel retrieval.

Depends on: PR-01, PR-03 optional.

Files:

- `backend/app/services/agentic/parallel_retrieval.py`

Tasks:

- [ ] Call existing `get_rag_service`.
- [ ] Use `query_deep` when available.
- [ ] Convert current chunks to `AgenticRetrievedChunk`.
- [ ] Deduplicate chunk IDs.
- [ ] Preserve image/table metadata.
- [ ] Add merge unit tests.

Review gates:

- [ ] No duplicated ChromaDB query logic.
- [ ] No changes to `DeepRetriever`.
- [ ] One sub-query failure does not kill all retrieval.

## PR-05: Analyzer And Planner

Goal: add query classification and output planning.

Depends on: PR-01.

Files:

- `backend/app/services/agentic/query_analyzer.py`
- `backend/app/services/agentic/response_planner.py`

Tasks:

- [ ] Implement LLM JSON calls.
- [ ] Add rule-based fallbacks.
- [ ] Add tests for VI and EN prompts.
- [ ] Add tests for multi-request split.

Review gates:

- [ ] Invalid LLM JSON is handled.
- [ ] Planner never returns empty `batch_now` for retrieval tasks.
- [ ] No provider-specific code outside LLM abstraction unless explicitly justified.

## PR-06: Sufficiency Judge And Query Rewriter

Goal: add retrieval feedback loop components.

Depends on: PR-01, PR-05.

Files:

- `backend/app/services/agentic/sufficiency_judge.py`
- `backend/app/services/agentic/query_rewriter.py`

Tasks:

- [ ] Implement structured sufficiency judgment.
- [ ] Detect missing evidence.
- [ ] Detect overloaded context.
- [ ] Implement expansion, step-back, and hyde rewrites.
- [ ] Add timeout/fallback behavior.

Review gates:

- [ ] Hard caps are respected by caller contract.
- [ ] Judge failure has safe fallback.
- [ ] Rewrite output remains retrieval-oriented, not final-answer-oriented.

## PR-07: Hierarchical Synthesizer And Response Judge

Goal: compact selected evidence and validate generated answers.

Depends on: PR-01, PR-03, PR-05.

Files:

- `backend/app/services/agentic/hierarchical_synthesizer.py`
- `backend/app/services/agentic/response_judge.py`

Tasks:

- [ ] Summarize chunks per sub-query.
- [ ] Preserve supporting chunk IDs.
- [ ] Assemble final context.
- [ ] Judge faithfulness.
- [ ] Judge completeness.
- [ ] Judge coverage against execution plan.

Review gates:

- [ ] Summary fallback exists.
- [ ] Judge failure does not produce 500.
- [ ] Citation mapping is preserved.

## PR-08: Continuation Manager

Goal: persist unfinished execution items.

Depends on: PR-01, PR-05.

Files:

- `backend/app/services/agentic/continuation_manager.py`
- `backend/app/models/agentic_session.py`
- `backend/alembic/versions/<revision>_add_agentic_sessions.py`
- `backend/app/models/__init__.py`

Tasks:

- [ ] Add model.
- [ ] Add migration.
- [ ] Save continuation state.
- [ ] Load active continuation state.
- [ ] Detect continuation intent.
- [ ] Expire old states.
- [ ] Add DB tests or repository-level tests.

Review gates:

- [ ] Migration upgrades and downgrades cleanly.
- [ ] State JSON has schema version.
- [ ] Continuation is workspace-scoped.

## PR-09: Orchestrator

Goal: integrate components into a controlled state machine without API changes.

Depends on: PR-01 through PR-08.

Files:

- `backend/app/services/agentic/orchestrator.py`

Tasks:

- [ ] Implement `run`.
- [ ] Implement `run_stream`.
- [ ] Enforce retrieval attempts.
- [ ] Enforce replan attempts.
- [ ] Add web search fallback branch.
- [ ] Add graceful fallback to existing Hybrid RAG response path.
- [ ] Add integration tests with mocked components.

Review gates:

- [ ] No API file changes in this PR.
- [ ] No parser/indexing code touched.
- [ ] Orchestrator can be tested with fake components.

## PR-10: API And SSE Integration

Goal: expose agentic orchestrator through existing endpoints.

Depends on: PR-09.

Files:

- `backend/app/api/rag.py`
- `backend/app/api/chat_agent.py`
- `backend/app/schemas/rag.py`

Tasks:

- [ ] Branch on `AGENTIC_RAG_ENABLED`.
- [ ] Preserve disabled behavior.
- [ ] Map orchestrator stream events to current SSE events.
- [ ] Add optional `agentic_metadata`.
- [ ] Preserve chat persistence.
- [ ] Preserve source and image response shape.

Review gates:

- [ ] Existing frontend works without changes.
- [ ] Agentic metadata is optional.
- [ ] Stream still sends `complete`.
- [ ] Error stream still sends `error`.

## PR-11: Observability And Evaluation

Goal: make agentic behavior measurable and debuggable.

Depends on: PR-10.

Files:

- `backend/scripts/agentic_eval_dataset.json`
- `backend/scripts/eval_agentic_rag.py`
- agentic component logs as needed

Tasks:

- [x] Add structured logs.
- [x] Add eval dataset.
- [x] Add eval script.
- [x] Add manual verification notes.

Review gates:

- [x] Logs do not leak secrets.
- [x] Eval script does not require web by default.
- [x] Debug output includes enough metadata to explain partial answers.

## Release Gate

Before enabling in shared dev:

- [ ] PR-01 through PR-11 merged.
- [ ] Agentic disabled path manually checked.
- [ ] Agentic enabled internal KB path manually checked.
- [ ] Web disabled path manually checked.
- [ ] Web enabled path checked with chosen provider.
- [ ] Continuation checked.
- [ ] UI stream checked.
- [ ] Static image rendering checked.
