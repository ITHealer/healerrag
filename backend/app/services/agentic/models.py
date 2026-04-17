"""Pydantic contracts for the Agentic RAG pipeline.

These models are intentionally provider-agnostic. They define the data passed
between future agentic components without changing the current Hybrid RAG path.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class QueryComplexity(str, Enum):
    """High-level retrieval complexity for a user request."""

    NO_RETRIEVAL = "no_retrieval"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"


class ChunkSource(str, Enum):
    """Origin of an evidence chunk."""

    VECTOR = "vector"
    KG = "kg"
    WEB = "web"


class RewriteStrategy(str, Enum):
    """Supported query rewrite strategies for retrieval retry."""

    EXPANSION = "expansion"
    STEP_BACK = "step_back"
    HYDE = "hyde"


class QueryAnalysisResult(BaseModel):
    """Structured output from the query analyzer."""

    complexity: QueryComplexity
    sub_queries: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    temporal_range: str | None = None
    language: str = "unknown"
    strategy_hint: str = ""
    reasoning: str = ""

    @model_validator(mode="after")
    def ensure_sub_queries_for_retrieval(self) -> "QueryAnalysisResult":
        if self.complexity != QueryComplexity.NO_RETRIEVAL and not self.sub_queries:
            raise ValueError("sub_queries is required for retrieval queries")
        return self


class ExecutionItem(BaseModel):
    """A user-facing work item that may be answered now or later."""

    item_id: str
    description: str
    priority: int = Field(default=1, ge=1)
    estimated_output_tokens: int = Field(default=300, ge=1)
    related_sub_queries: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Response planner decision for current-turn work and continuation."""

    can_fully_answer_now: bool = True
    total_items: int = 0
    items: list[ExecutionItem] = Field(default_factory=list)
    batch_now: list[str] = Field(default_factory=list)
    batch_later: list[str] = Field(default_factory=list)
    continuation_message: str | None = None
    reasoning: str = ""

    @model_validator(mode="after")
    def validate_batch_item_ids(self) -> "ExecutionPlan":
        item_ids = {item.item_id for item in self.items}
        selected_ids = set(self.batch_now) | set(self.batch_later)
        unknown_ids = selected_ids - item_ids
        if unknown_ids:
            raise ValueError(f"batch item IDs are not present in items: {sorted(unknown_ids)}")

        if self.items and self.total_items == 0:
            self.total_items = len(self.items)
        return self


class AgenticRetrievedChunk(BaseModel):
    """A normalized evidence chunk used by Agentic RAG components."""

    chunk_id: str
    content: str
    score: float = 0.0
    source: ChunkSource
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "content")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class WebSearchSource(BaseModel):
    """One cited web source returned by a provider-backed search."""

    url: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchResult(BaseModel):
    """Search answer and citations for one normalized query."""

    query: str
    answer: str = ""
    sources: list[WebSearchSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    provider: str
    model: str


class WebSearchOutput(BaseModel):
    """Provider-backed web search output before evidence chunk conversion."""

    backend: str
    model: str
    results: list[WebSearchResult] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Evidence returned for one sub-query."""

    sub_query: str
    chunks: list[AgenticRetrievedChunk] = Field(default_factory=list)
    reranked: bool = False
    error_message: str | None = None


class SufficiencyJudgment(BaseModel):
    """Decision on whether current evidence is enough to answer."""

    is_sufficient: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    covered_aspects: list[str] = Field(default_factory=list)
    suggested_rewrite_strategy: RewriteStrategy | str | None = None
    reasoning: str = ""


class RewrittenQuery(BaseModel):
    """A retrieval retry query produced from missing evidence."""

    original_query: str
    rewritten_query: str
    strategy: RewriteStrategy
    iteration: int = Field(ge=1)


class ContextBudgetDecision(BaseModel):
    """Selected evidence after final-context budget enforcement."""

    max_final_context_tokens: int = Field(ge=1)
    max_final_chunks: int = Field(ge=1)
    max_chunks_per_subquery: int = Field(ge=1)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="after")
    def ensure_no_selected_dropped_overlap(self) -> "ContextBudgetDecision":
        overlap = set(self.selected_chunk_ids) & set(self.dropped_chunk_ids)
        if overlap:
            raise ValueError(f"chunks cannot be both selected and dropped: {sorted(overlap)}")
        return self


class SubQuerySummary(BaseModel):
    """Compact summary for one sub-query with citation mapping."""

    sub_query: str
    summary: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class ResponseJudgment(BaseModel):
    """Quality judgment for the generated answer."""

    pass_judge: bool
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ContinuationState(BaseModel):
    """Persistable state for unfinished execution items."""

    schema_version: int = 1
    session_id: str
    original_query: str
    execution_plan: ExecutionPlan
    completed_item_ids: list[str] = Field(default_factory=list)
    remaining_item_ids: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class AgenticRAGState(BaseModel):
    """Full orchestration state for one Agentic RAG run."""

    original_query: str
    workspace_id: str
    session_id: str | None = None

    analysis: QueryAnalysisResult | None = None
    execution_plan: ExecutionPlan | None = None

    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    merged_chunks: list[AgenticRetrievedChunk] = Field(default_factory=list)

    sufficiency: SufficiencyJudgment | None = None
    rewrite_history: list[RewrittenQuery] = Field(default_factory=list)
    retrieval_attempts: int = Field(default=0, ge=0)

    context_budget: ContextBudgetDecision | None = None
    subquery_summaries: list[SubQuerySummary] = Field(default_factory=list)
    assembled_context: str = ""

    generated_answer: str = ""
    response_judgment: ResponseJudgment | None = None
    replan_attempts: int = Field(default=0, ge=0)

    continuation_state: ContinuationState | None = None
    continuation_offered: bool = False

    final_answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sources_used: list[ChunkSource] = Field(default_factory=list)
