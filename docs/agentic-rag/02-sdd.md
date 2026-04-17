# SDD: HealerRAG Agentic RAG Backend Design

Version: 1.0
Date: 2026-04-15
Status: Draft

## 1. Architecture Summary

The Agentic RAG upgrade adds a new orchestration layer around the current retrieval system.

The current indexing stack remains unchanged:

```text
Upload -> Parser -> Chunk Dedup -> Embedding -> ChromaDB
                         |
                         -> LightRAG KG ingest
```

The new query stack becomes:

```text
User Query
-> QueryAnalyzer
-> ResponsePlanner
-> ParallelRetrievalCoordinator
-> SufficiencyJudge
-> QueryRewriter or WebSearchTool when needed
-> ContextBudgetManager
-> HierarchicalSynthesizer
-> LLM generation
-> ResponseJudge
-> ContinuationManager
-> API response or SSE stream
```

## 2. Package Layout

Create:

```text
backend/app/services/agentic/
  __init__.py
  models.py
  prompts.py
  query_analyzer.py
  response_planner.py
  parallel_retrieval.py
  sufficiency_judge.py
  query_rewriter.py
  context_budget_manager.py
  hierarchical_synthesizer.py
  response_judge.py
  continuation_manager.py
  web_search_tool.py
  orchestrator.py
```

Optional later:

```text
backend/app/models/agentic_session.py
backend/alembic/versions/<revision>_add_agentic_sessions.py
```

## 3. Integration Points

### 3.1 API Layer

`backend/app/api/rag.py`

- Keep current endpoint paths.
- If `settings.AGENTIC_RAG_ENABLED` is false, use existing logic.
- If enabled, call `AgenticRAGOrchestrator`.
- Return existing response fields plus optional `agentic_metadata`.

`backend/app/api/chat_agent.py`

- Keep SSE event names.
- Add adapter from orchestrator stream events to current SSE events.
- Do not move persistence logic into orchestrator.

### 3.2 Current Retrieval Layer

`backend/app/services/agentic_rag_service.py`

- Do not rename.
- Continue to own document process/index behavior.
- Continue to expose `query_deep()`.
- The new `ParallelRetrievalCoordinator` should call this service rather than duplicating retrieval internals.

`backend/app/services/deep_retriever.py`

- Continue to own hybrid retrieval and reranking.
- Do not add agentic orchestration loops here.

### 3.3 LLM Provider Layer

`backend/app/services/llm/__init__.py`

- Continue to provide `get_llm_provider()` and `get_embedding_provider()`.
- Agentic components should use this abstraction for analyzer, planner, judge, and synthesis unless a component-specific model is configured later.

## 4. Data Models

All models live in `services/agentic/models.py`.

### 4.1 Enums

```python
class QueryComplexity(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"

class ChunkSource(str, Enum):
    VECTOR = "vector"
    KG = "kg"
    WEB = "web"

class RewriteStrategy(str, Enum):
    EXPANSION = "expansion"
    STEP_BACK = "step_back"
    HYDE = "hyde"
```

### 4.2 Core Models

Use `AgenticRetrievedChunk` instead of `RetrievedChunk`.

Required fields:

- `chunk_id: str`
- `content: str`
- `score: float`
- `source: ChunkSource`
- `metadata: dict`

Metadata conventions:

- `document_id`
- `page_no`
- `heading_path`
- `source_file`
- `covered_sub_query`
- `covered_sub_queries`
- `covered_entities`
- `retrieval_attempt`
- `image_refs`
- `table_refs`
- `web_url`
- `web_title`
- `web_provider`

## 5. Component Contracts

### 5.1 QueryAnalyzer

File: `query_analyzer.py`

Public method:

```python
async def analyze(self, query: str, history: list[dict] | None = None) -> QueryAnalysisResult
```

Fallback:

- If query is greeting or chitchat, return `NO_RETRIEVAL`.
- If query contains conjunctions or multiple clauses, likely `MULTI_HOP`.
- If query contains `and`, `và`, or comma-separated tasks, mark strategy hint as multi-request.

### 5.2 ResponsePlanner

File: `response_planner.py`

Public method:

```python
async def plan(self, query: str, analysis: QueryAnalysisResult) -> ExecutionPlan
```

Rules:

- One clear task becomes one execution item.
- Multiple requested analyses become multiple execution items.
- Estimate output tokens by task type.
- If total estimate exceeds config, split into `batch_now` and `batch_later`.
- Always prioritize tasks that directly answer the main query.

### 5.3 ParallelRetrievalCoordinator

File: `parallel_retrieval.py`

Public methods:

```python
async def retrieve_all(...) -> list[RetrievalResult]
def merge_results(results: list[RetrievalResult]) -> list[AgenticRetrievedChunk]
```

Rules:

- Use `get_rag_service(db, workspace_id)`.
- If returned service supports `query_deep`, call it.
- Use `asyncio.gather` with timeout.
- Convert existing chunk/citation objects into `AgenticRetrievedChunk`.
- Deduplicate by stable `chunk_id`.

### 5.4 SufficiencyJudge

File: `sufficiency_judge.py`

Public method:

```python
async def judge(
    self,
    original_query: str,
    analysis: QueryAnalysisResult,
    execution_plan: ExecutionPlan,
    chunks: list[AgenticRetrievedChunk],
) -> SufficiencyJudgment
```

Output must be structured JSON.

Fallback:

- If judge call fails and chunks exist, return sufficient with low confidence.
- If no chunks exist, return insufficient.

### 5.5 QueryRewriter

File: `query_rewriter.py`

Public method:

```python
async def rewrite(
    self,
    original_query: str,
    missing_aspects: list[str],
    strategy: RewriteStrategy,
    iteration: int,
) -> RewrittenQuery
```

Rules:

- `expansion`: add missing entities, synonyms, time range.
- `step_back`: broaden query to parent concept.
- `hyde`: generate hypothetical answer text for retrieval only.

### 5.6 WebSearchTool

File: `web_search_tool.py`

Public method:

```python
async def search(self, query: str | None = None, queries: list[str] | None = None) -> WebSearchOutput
```

Backend support:

- OpenAI Responses API with `tools=[{"type": "web_search"}]`.
- Google GenAI with `Tool(google_search=GoogleSearch())`.

Credential rules:

- OpenAI uses `OPENAI_API_KEY`.
- Google uses `GOOGLE_AI_API_KEY` or `GEMINI_API_KEY`.
- Missing credentials produce controlled exception.

Security:

- Do not run if `AGENTIC_WEB_SEARCH_ENABLED=false`.
- Cap query count.
- Cap timeout.
- Store URL/title only, not full web page HTML.

### 5.7 ContextBudgetManager

File: `context_budget_manager.py`

Public method:

```python
def select(
    self,
    chunks: list[AgenticRetrievedChunk],
    sub_queries: list[str],
) -> ContextBudgetDecision
```

No LLM call in first release.

Selection algorithm:

1. Group chunks by covered sub-query.
2. Pick best chunk per sub-query.
3. Fill remaining slots by score.
4. Prefer source diversity.
5. Drop overflow by lowest priority.

### 5.8 HierarchicalSynthesizer

File: `hierarchical_synthesizer.py`

Public methods:

```python
async def summarize(...) -> list[SubQuerySummary]
def assemble(...) -> str
```

Rules:

- Summaries must stay grounded in selected chunks.
- Preserve `supporting_chunk_ids`.
- If LLM fails, use compact raw evidence fallback.

### 5.9 ResponseJudge

File: `response_judge.py`

Public method:

```python
async def judge(
    self,
    original_query: str,
    generated_answer: str,
    chunks: list[AgenticRetrievedChunk],
    execution_plan: ExecutionPlan,
) -> ResponseJudgment
```

Rules:

- Fail if answer uses claims unsupported by selected evidence.
- Fail if answer misses required `batch_now` items.
- Pass if minor formatting issues only.

### 5.10 ContinuationManager

File: `continuation_manager.py`

Public methods:

```python
async def save(...) -> ContinuationState
async def load_active(...) -> ContinuationState | None
async def mark_completed(...) -> ContinuationState
def is_continuation_request(message: str) -> bool
```

Storage:

- Use `agentic_sessions` table.
- Store full state as JSON.
- Scope by workspace and session.

### 5.11 AgenticRAGOrchestrator

File: `orchestrator.py`

Public methods:

```python
async def run(...) -> AgenticRAGState
async def run_stream(...) -> AsyncGenerator[AgenticRAGEvent, None]
```

Responsibilities:

- Own state transitions.
- Enforce attempt caps.
- Call components in order.
- Degrade gracefully.
- Never perform document parsing or indexing.

## 6. Web Search Design

The web search tool must follow provider-backed search.

### 6.1 OpenAI Backend

Use `AsyncOpenAI.responses.create` with:

- configured model
- `tools=[{"type": "web_search"}]`
- `temperature=0`

Extract:

- response text
- URL citations
- titles

### 6.2 Google Backend

Use `google.genai.Client` with:

- configured model
- `GenerateContentConfig`
- `Tool(google_search=GoogleSearch())`

Extract:

- response text
- grounding chunks
- web search queries

### 6.3 Conversion To Evidence Chunks

Each web result becomes one or more `AgenticRetrievedChunk` objects:

- `source=ChunkSource.WEB`
- `chunk_id="web_<provider>_<stable_hash>"`
- `content=answer`
- `metadata.url`
- `metadata.title`
- `metadata.provider`
- `metadata.model`

Web chunks can support answers but must be cited separately from document chunks.

## 7. SSE Event Mapping

Internal agentic event names may be richer, but API must emit existing event names.

Mapping:

- `analysis_started`, `analysis_done` -> `status`
- `planning_done` -> `status`
- `retrieval_started`, `retrieval_retry` -> `status`
- `sources_selected` -> `sources`
- `images_selected` -> `images`
- `generation_token` -> `token`
- `run_complete` -> `complete`
- `run_error` -> `error`

Do not remove current event types.

## 8. Error Handling

Every agentic component must return controlled fallback behavior.

Required fallbacks:

- QueryAnalyzer failure -> rule-based analysis.
- ResponsePlanner failure -> single execution item.
- Retrieval failure for one sub-query -> continue with other sub-queries.
- SufficiencyJudge failure -> continue if chunks exist.
- WebSearchTool failure -> log and continue without web.
- HierarchicalSynthesizer failure -> raw selected chunks.
- ResponseJudge failure -> return generated answer with warning metadata.
- ContinuationManager failure -> return answer without continuation persistence.

## 9. Logging

Use existing Python logger pattern.

Required log keys in messages or extra metadata:

- `agentic_run_id`
- `workspace_id`
- `session_id`
- `complexity`
- `sub_query_count`
- `execution_item_count`
- `retrieval_attempt`
- `rewrite_strategy`
- `web_search_backend`
- `selected_chunk_count`
- `dropped_chunk_count`
- `response_judge_pass`
- `continuation_offered`

## 10. Testing Strategy

Unit tests first:

- models validate
- planner splits correctly
- context budget deterministic
- merge dedup works
- web search extraction works with fixture raw responses
- continuation state serializes/deserializes

Integration tests after:

- agentic disabled path
- agentic enabled simple internal query
- agentic enabled multi-hop query
- agentic enabled multi-request continuation
- web disabled no external call
- web enabled missing credentials controlled error

## 11. Deployment

Default deploy config:

```env
AGENTIC_RAG_ENABLED=false
AGENTIC_WEB_SEARCH_ENABLED=false
```

Enable locally first.

Only enable web search after credentials and provider model are verified.
