import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.core.config import Settings
from app.services.agentic import (
    AgenticRAGState,
    AgenticRetrievedChunk,
    ChunkSource,
    ContextBudgetDecision,
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
)
from app.services.agentic.prompts import (
    QUERY_ANALYZER_SYSTEM_PROMPT,
    RESPONSE_JUDGE_SYSTEM_PROMPT,
    RESPONSE_PLANNER_SYSTEM_PROMPT,
)


class AgenticModelTests(unittest.TestCase):
    def test_query_analysis_requires_sub_queries_for_retrieval(self) -> None:
        with self.assertRaises(ValidationError):
            QueryAnalysisResult(complexity=QueryComplexity.SINGLE_HOP)

        analysis = QueryAnalysisResult(
            complexity=QueryComplexity.SINGLE_HOP,
            sub_queries=["How do I login?"],
            entities=["login"],
            language="en",
        )

        self.assertEqual(analysis.complexity, QueryComplexity.SINGLE_HOP)
        self.assertEqual(analysis.sub_queries, ["How do I login?"])

    def test_execution_plan_validates_batch_ids(self) -> None:
        item = ExecutionItem(
            item_id="item_1",
            description="Explain the login flow",
            priority=1,
            estimated_output_tokens=250,
            related_sub_queries=["login flow"],
        )

        plan = ExecutionPlan(items=[item], batch_now=["item_1"])

        self.assertEqual(plan.total_items, 1)
        self.assertEqual(plan.batch_now, ["item_1"])

        with self.assertRaises(ValidationError):
            ExecutionPlan(items=[item], batch_now=["missing_item"])

    def test_retrieval_and_state_models_round_trip(self) -> None:
        chunk = AgenticRetrievedChunk(
            chunk_id="chunk_1",
            content="Login requires a username and password.",
            score=0.9,
            source=ChunkSource.VECTOR,
            metadata={"covered_sub_query": "login"},
        )
        retrieval = RetrievalResult(sub_query="login", chunks=[chunk], reranked=True)
        budget = ContextBudgetDecision(
            max_final_context_tokens=5000,
            max_final_chunks=8,
            max_chunks_per_subquery=2,
            selected_chunk_ids=["chunk_1"],
            dropped_chunk_ids=[],
        )
        summary = SubQuerySummary(
            sub_query="login",
            summary="The document says login uses username and password.",
            supporting_chunk_ids=["chunk_1"],
        )
        judgment = SufficiencyJudgment(
            is_sufficient=True,
            confidence=0.85,
            covered_aspects=["login credential requirement"],
        )
        rewrite = RewrittenQuery(
            original_query="login",
            rewritten_query="login username password",
            strategy=RewriteStrategy.EXPANSION,
            iteration=1,
        )
        response_judgment = ResponseJudgment(
            pass_judge=True,
            faithfulness_score=0.9,
            completeness_score=0.8,
        )

        state = AgenticRAGState(
            original_query="How do I login?",
            workspace_id="1",
            retrieval_results=[retrieval],
            merged_chunks=[chunk],
            sufficiency=judgment,
            rewrite_history=[rewrite],
            context_budget=budget,
            subquery_summaries=[summary],
            response_judgment=response_judgment,
            final_answer="Use your username and password.",
            sources_used=[ChunkSource.VECTOR],
        )

        dumped = state.model_dump(mode="json")

        self.assertEqual(dumped["sources_used"], ["vector"])
        self.assertEqual(dumped["retrieval_results"][0]["chunks"][0]["chunk_id"], "chunk_1")
        self.assertEqual(dumped["context_budget"]["selected_chunk_ids"], ["chunk_1"])

    def test_agentic_config_defaults_are_safe(self) -> None:
        clean_env = {
            "AGENTIC_RAG_ENABLED": "",
            "AGENTIC_WEB_SEARCH_ENABLED": "",
            "WEB_SEARCH_TOOL_BACKEND": "",
        }
        with patch.dict("os.environ", clean_env, clear=False):
            for key in clean_env:
                del __import__("os").environ[key]

            settings = Settings(_env_file=None)

        self.assertIs(settings.AGENTIC_RAG_ENABLED, False)
        self.assertIs(settings.AGENTIC_WEB_SEARCH_ENABLED, False)
        self.assertEqual(settings.WEB_SEARCH_TOOL_BACKEND, "auto")
        self.assertEqual(settings.AGENTIC_RAG_MAX_RETRIEVAL_ATTEMPTS, 3)
        self.assertEqual(settings.AGENTIC_MAX_FINAL_CHUNKS, 8)
        self.assertEqual(settings.WEB_SEARCH_MAX_QUERIES, 2)

    def test_prompt_constants_are_present(self) -> None:
        self.assertIn("QueryAnalysisResult", QUERY_ANALYZER_SYSTEM_PROMPT)
        self.assertIn("ExecutionPlan", RESPONSE_PLANNER_SYSTEM_PROMPT)
        self.assertIn("ResponseJudgment", RESPONSE_JUDGE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
