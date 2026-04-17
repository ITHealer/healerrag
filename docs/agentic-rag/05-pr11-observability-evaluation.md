# PR-11 Observability And Evaluation Notes

Date: 2026-04-15
Status: Implemented

## Scope

PR-11 adds backend-only observability and offline evaluation for the Agentic RAG rollout.

Implemented files:

- `backend/app/services/agentic/observability.py`
- `backend/scripts/agentic_eval_dataset.json`
- `backend/scripts/eval_agentic_rag.py`
- `backend/tests/agentic/test_observability_eval.py`

Integrated files:

- `backend/app/services/agentic/orchestrator.py`
- `backend/app/services/agentic/__init__.py`
- `backend/app/api/rag.py`

## Observability Coverage

Structured logs now include safe, content-free metadata:

- `agentic_run_id`
- `workspace_id`
- `session_id`
- `complexity`
- `language`
- `sub_query_count`
- `execution_item_count`
- `batch_now_count`
- `batch_later_count`
- `retrieval_attempts`
- `rewrite_count`
- `rewrite_strategies`
- `web_search_used`
- `selected_chunk_ids`
- `dropped_chunk_ids`
- `response_judge_pass`
- `continuation_offered`
- `remaining_items_count`

Logs intentionally do not include raw query text, raw chunk text, API keys, or provider credentials.

## Agentic Metadata

API responses reuse `state_observability_metadata()` so `/rag/query`, `/rag/chat`, and SSE `complete` events can explain:

- why a response was partial
- how many retrieval attempts ran
- whether rewrite or web search was used
- which chunks were selected or dropped by the context budget
- whether continuation was offered

## Offline Evaluation

The default eval script is offline and rule-based:

```powershell
cd backend
uv run python scripts/eval_agentic_rag.py
```

It does not require:

- database
- ChromaDB
- web search
- external LLM provider
- API credentials

The dataset contains 70 cases:

- 15 no-retrieval
- 20 single-hop
- 20 multi-hop
- 15 multi-request

The script writes results to:

- `backend/scripts/agentic_eval_results.json`

## Verification Checklist

- Compile backend and tests.
- Run unit tests.
- Run offline agentic eval script.
- Confirm `AGENTIC_WEB_SEARCH_ENABLED=false` remains compatible with eval.
- Confirm debug metadata contains counts and IDs, not content.
