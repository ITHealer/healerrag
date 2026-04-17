"""Agentic RAG foundation package.

This package contains type contracts and prompt constants for the opt-in
Agentic RAG pipeline. Runtime integration is intentionally added in later PRs.
"""

from app.services.agentic.models import (
    AgenticRAGState,
    AgenticRetrievedChunk,
    ChunkSource,
    ContextBudgetDecision,
    ContinuationState,
    ExecutionItem,
    ExecutionPlan,
    QueryAnalysisResult,
    QueryComplexity,
    ResponseJudgment,
    RetrievalResult,
    RewriteStrategy,
    RewrittenQuery,
    SubQuerySummary,
    SufficiencyJudgment,
    WebSearchOutput,
    WebSearchResult,
    WebSearchSource,
)
from app.services.agentic.context_budget_manager import ContextBudgetManager
from app.services.agentic.continuation_manager import ContinuationManager
from app.services.agentic.hierarchical_synthesizer import HierarchicalSynthesizer
from app.services.agentic.observability import (
    agentic_log_extra,
    log_agentic_event,
    state_observability_metadata,
)
from app.services.agentic.orchestrator import AgenticRAGOrchestrator
from app.services.agentic.parallel_retrieval import ParallelRetrievalCoordinator
from app.services.agentic.query_analyzer import QueryAnalyzer
from app.services.agentic.query_rewriter import QueryRewriter
from app.services.agentic.response_judge import ResponseJudge
from app.services.agentic.response_planner import ResponsePlanner
from app.services.agentic.sufficiency_judge import SufficiencyJudge
from app.services.agentic.web_search_tool import WebSearchTool

__all__ = [
    "AgenticRAGState",
    "AgenticRetrievedChunk",
    "AgenticRAGOrchestrator",
    "agentic_log_extra",
    "ChunkSource",
    "ContextBudgetDecision",
    "ContextBudgetManager",
    "ContinuationManager",
    "ContinuationState",
    "ExecutionItem",
    "ExecutionPlan",
    "HierarchicalSynthesizer",
    "log_agentic_event",
    "QueryAnalysisResult",
    "QueryAnalyzer",
    "QueryComplexity",
    "QueryRewriter",
    "ParallelRetrievalCoordinator",
    "ResponseJudgment",
    "ResponseJudge",
    "ResponsePlanner",
    "RetrievalResult",
    "RewriteStrategy",
    "RewrittenQuery",
    "SubQuerySummary",
    "SufficiencyJudgment",
    "SufficiencyJudge",
    "state_observability_metadata",
    "WebSearchOutput",
    "WebSearchResult",
    "WebSearchSource",
    "WebSearchTool",
]
