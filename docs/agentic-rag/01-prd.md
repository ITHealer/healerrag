# PRD: HealerRAG Agentic RAG Upgrade

Version: 1.0
Date: 2026-04-15
Status: Draft
Owner: Backend

## 1. Background

HealerRAG currently uses a Hybrid RAG pipeline:

- Document text is parsed into markdown and chunks.
- Chunks are embedded and stored in ChromaDB.
- Markdown is optionally ingested into LightRAG Knowledge Graph.
- Query uses vector retrieval plus KG context and reranking.
- Chat can stream answer tokens, sources, images, and thinking metadata.

This is stable and must remain the fallback path.

The limitation is that current retrieval is mostly single-pass. For complex questions, broad time ranges, multi-hop reasoning, or multi-request prompts, the system retrieves once and generates once. It cannot reliably decide whether context is sufficient, retry with better queries, trim overloaded context, or persist remaining work for continuation.

## 2. Goals

G1. Add optional Agentic RAG orchestration while preserving the current API.

G2. Classify user requests into:

- `no_retrieval`
- `single_hop`
- `multi_hop`

G3. Detect multi-request prompts and split answer work into execution items.

G4. Decompose complex queries into sub-queries.

G5. Retrieve evidence in parallel for sub-queries.

G6. Judge whether retrieved evidence is sufficient before generation.

G7. Rewrite queries when evidence is insufficient.

G8. Add provider-backed web search as an optional fallback.

G9. Manage final context budget before generation.

G10. Summarize evidence hierarchically by sub-query.

G11. Judge generated response for faithfulness, completeness, and plan coverage.

G12. Support continuation when a user asks for too much in one turn.

G13. Keep current frontend usable without required React changes.

## 3. Non-Goals

This work must not:

- Replace Docling or Marker parser.
- Replace ChromaDB.
- Replace LightRAG.
- Change existing chunk metadata shape in ChromaDB.
- Train a custom model.
- Introduce LangGraph.
- Require frontend rewrite.
- Require web search for normal internal-document questions.

## 4. User Stories

### 4.1 Simple Internal Question

As a user, I ask: "Cách login vào BESTmed?"

Expected:

- System retrieves relevant document chunks.
- System returns concise answer with citations.
- Images still appear when relevant.
- No unnecessary multi-step agentic overhead is visible.

### 4.2 Complex Multi-hop Question

As a user, I ask: "Tóm tắt các bước login, forgot password, và lỗi account locked trong tài liệu."

Expected:

- System identifies multiple sub-queries.
- System retrieves evidence for each topic.
- System keeps coverage across all sub-queries.
- Final answer is organized by topic.

### 4.3 Multi-request Question

As a user, I ask for five or more independent analysis tasks in one message.

Expected:

- System creates an execution plan.
- System answers the highest-priority subset now if output would be too long.
- System clearly tells the user what was completed.
- System saves remaining items for continuation.

### 4.4 Fresh Information Question

As a user, I ask a question requiring up-to-date public information not present in the KB.

Expected:

- Internal KB retrieval runs first.
- Sufficiency judge detects missing evidence.
- If `AGENTIC_WEB_SEARCH_ENABLED=true`, web search may run.
- Web evidence is clearly marked as web evidence.
- If web search is disabled, system explains that the documents do not contain enough current information.

## 5. Functional Requirements

### FR-01 Feature Flags

Add:

```env
AGENTIC_RAG_ENABLED=false
AGENTIC_WEB_SEARCH_ENABLED=false
```

All agentic behavior must depend on `AGENTIC_RAG_ENABLED`.

All web calls must depend on both:

- `AGENTIC_RAG_ENABLED=true`
- `AGENTIC_WEB_SEARCH_ENABLED=true`

### FR-02 Backward Compatibility

Existing endpoints must remain:

- `POST /api/v1/rag/query/{workspace_id}`
- `POST /api/v1/rag/chat/{workspace_id}`
- `POST /api/v1/rag/chat/{workspace_id}/stream`

Existing response fields must remain valid.

New metadata must be optional.

### FR-03 Query Analysis

System must produce structured analysis:

- complexity
- sub_queries
- entities
- temporal_range
- language
- strategy_hint
- reasoning

If LLM analyzer fails, system must fall back to rule-based analysis.

### FR-04 Response Planning

System must create an execution plan for retrieval queries:

- execution items
- priority
- output token estimates
- batch_now
- batch_later
- continuation message

If all items fit, `batch_later` must be empty.

### FR-05 Parallel Retrieval

For each active sub-query, system must retrieve using current internal retrieval:

- vector search through ChromaDB
- KG raw context through LightRAG
- reranker through existing `RerankerService`

The coordinator must deduplicate chunks across sub-queries.

### FR-06 Sufficiency Judge

System must judge whether current evidence can answer `batch_now`.

Judge output must include:

- sufficient flag
- confidence
- covered_aspects
- missing_aspects
- suggested rewrite strategy
- reasoning

### FR-07 Query Rewrite

If insufficient and attempts remain, system may rewrite query using:

- expansion
- step_back
- hyde

The system must stop at configured max attempts.

### FR-08 Web Search Fallback

Web search must be provider-backed.

Supported backends:

- Google Gemini grounding search.
- OpenAI Responses web search.

Backend selection:

- explicit `WEB_SEARCH_TOOL_BACKEND`
- else `auto`
- `auto` prefers Google if Google key is present, then OpenAI.

Web search result must include:

- provider
- model
- query
- answer
- sources
- search_queries

### FR-09 Context Budget

Before final generation, system must select evidence under budget:

- max final context tokens
- max final chunks
- max chunks per sub-query

Each sub-query with available evidence should keep at least one strong chunk.

### FR-10 Hierarchical Synthesis

System must summarize selected evidence by sub-query before final generation.

Each sub-query summary must retain supporting chunk IDs.

If summarization fails, system must fallback to raw selected chunks.

### FR-11 Response Judge

System must judge answer after generation:

- faithfulness
- completeness
- coverage of `batch_now`

If judge fails and replan attempts remain, system can retry.

If judge repeatedly fails, system must return best effort answer with transparent caveat rather than 500.

### FR-12 Continuation

If `batch_later` is not empty:

- save continuation state
- append continuation offer to the answer
- allow user to resume with natural text such as "tiếp tục"

Continuation state must be scoped to workspace and session.

## 6. Non-Functional Requirements

### NFR-01 Performance

Simple query should not become significantly slower when agentic mode is disabled.

When enabled:

- retrieval tasks may run in parallel
- LLM judge calls must have timeout
- max attempts must be enforced

### NFR-02 Reliability

Agentic errors must degrade gracefully:

- analyzer fail: rule-based fallback
- sufficiency judge fail: continue with retrieved evidence
- web search fail: continue without web evidence
- response judge fail: return best effort
- continuation save fail: answer still returns

### NFR-03 Observability

Backend logs must include:

- agentic_run_id
- workspace_id
- complexity
- sub_query_count
- retrieval_attempts
- rewrite_count
- selected_chunk_ids
- dropped_chunk_ids
- web_search_used
- continuation_offered

### NFR-04 Maintainability

Each agentic component must be small, single-purpose, and independently testable.

API files should remain thin adapters.

## 7. Success Criteria

The feature is complete when:

- Agentic mode can be toggled by config.
- Existing UI still works with no required frontend changes.
- Simple document questions still return sources and images.
- Multi-hop questions run sub-query retrieval.
- Context budget prevents raw context overload.
- Multi-request prompts can produce continuation.
- Web search can be enabled separately and returns provider-backed sources.
- All new components have tests or clear manual verification steps.

## 8. Out of Scope For First Release

- Frontend UI for editing continuation queue.
- Admin dashboard for agentic metrics.
- Persistent cache for all LLM judge outputs.
- Per-user auth or user-specific sessions.
- LangGraph or other external workflow engine.
