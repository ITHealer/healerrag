import unittest

from app.services.agentic.models import (
    AgenticRetrievedChunk,
    ChunkSource,
    ExecutionItem,
    ExecutionPlan,
    QueryAnalysisResult,
    QueryComplexity,
    ResponseJudgment,
    RetrievalResult,
    RewriteStrategy,
    RewrittenQuery,
    SufficiencyJudgment,
    SubQuerySummary,
)
from app.services.agentic.orchestrator import AgenticRAGOrchestrator
from app.services.llm.types import LLMResult, StreamChunk


class FakeAnalyzer:
    async def analyze(self, query, history=None):
        return QueryAnalysisResult(
            complexity=QueryComplexity.SINGLE_HOP,
            sub_queries=[query],
            entities=["NVDA"],
            language="en",
        )


class FakePlanner:
    async def plan(self, *, query, analysis):
        return ExecutionPlan(
            can_fully_answer_now=True,
            total_items=1,
            items=[
                ExecutionItem(
                    item_id="item_1",
                    description=query,
                    priority=1,
                    estimated_output_tokens=300,
                    related_sub_queries=analysis.sub_queries,
                )
            ],
            batch_now=["item_1"],
            batch_later=[],
        )


class FakeRetriever:
    def __init__(self):
        self.calls: list[list[str]] = []

    async def retrieve_all(self, **kwargs):
        sub_queries = list(kwargs["sub_queries"])
        self.calls.append(sub_queries)
        chunks = [
            AgenticRetrievedChunk(
                chunk_id=f"c{len(self.calls)}",
                content=f"Evidence for {sub_queries[0]}",
                score=0.9,
                source=ChunkSource.VECTOR,
                metadata={
                    "document_id": 1,
                    "page_no": 2,
                    "chunk_index": len(self.calls),
                    "heading_path": ["Section"],
                    "covered_sub_query": sub_queries[0],
                    "covered_sub_queries": sub_queries,
                },
            )
        ]
        return [RetrievalResult(sub_query=sub_queries[0], chunks=chunks, reranked=True)]

    def merge_results(self, results):
        chunks = []
        for result in results:
            chunks.extend(result.chunks)
        return chunks


class FakeSufficiencyJudge:
    def __init__(self, judgments):
        self.judgments = list(judgments)

    async def judge(self, **kwargs):
        if self.judgments:
            return self.judgments.pop(0)
        return SufficiencyJudgment(is_sufficient=True, confidence=1.0)


class FakeRewriter:
    def __init__(self):
        self.calls = []

    async def rewrite(self, *, original_query, missing_aspects, strategy, iteration):
        self.calls.append((strategy, iteration, list(missing_aspects)))
        return RewrittenQuery(
            original_query=original_query,
            rewritten_query=f"{original_query} expanded",
            strategy=RewriteStrategy.EXPANSION,
            iteration=iteration,
        )


class FakeSynthesizer:
    async def summarize(self, *, chunks, sub_queries):
        return [
            SubQuerySummary(
                sub_query=sub_queries[0] if sub_queries else "query",
                summary="Grounded summary.",
                supporting_chunk_ids=[chunk.chunk_id for chunk in chunks],
            )
        ]

    def assemble(self, *, original_query, summaries, chunks):
        return "Compact context from selected chunks."


class FakeResponseJudge:
    async def judge(self, **kwargs):
        return ResponseJudgment(
            pass_judge=True,
            faithfulness_score=1.0,
            completeness_score=1.0,
            issues=[],
            reasoning="ok",
        )


class FakeContinuationManager:
    def is_continuation_intent(self, message):
        return False

    async def load_active(self, **kwargs):
        return None

    async def save(self, **kwargs):
        return None

    async def mark_completed(self, **kwargs):
        return True


class FakeProvider:
    async def acomplete(self, messages, **kwargs):
        return LLMResult(content="Final grounded answer.")

    async def astream(self, messages, **kwargs):
        yield StreamChunk(type="text", text="Final ")
        yield StreamChunk(type="text", text="grounded answer.")


class FakeWebSearchTool:
    def __init__(self):
        self.calls = 0

    async def search(self, **kwargs):
        self.calls += 1
        raise AssertionError("web search should not be called")


class AgenticRAGOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self, *, sufficiency_judgments, web_search_enabled=False, web_search_tool=None):
        return AgenticRAGOrchestrator(
            query_analyzer=FakeAnalyzer(),
            response_planner=FakePlanner(),
            parallel_retriever=FakeRetriever(),
            sufficiency_judge=FakeSufficiencyJudge(sufficiency_judgments),
            query_rewriter=FakeRewriter(),
            hierarchical_synthesizer=FakeSynthesizer(),
            response_judge=FakeResponseJudge(),
            continuation_manager=FakeContinuationManager(),
            web_search_tool=web_search_tool,
            llm_provider=FakeProvider(),
            max_retrieval_attempts=2,
            max_replan_attempts=0,
            web_search_enabled=web_search_enabled,
        )

    async def test_run_completes_with_selected_chunks_and_metadata(self) -> None:
        orchestrator = self._orchestrator(
            sufficiency_judgments=[SufficiencyJudgment(is_sufficient=True, confidence=0.9)]
        )

        state = await orchestrator.run(
            query="NVDA revenue 2023",
            workspace_id=1,
            db=object(),
            session_id="s1",
        )

        self.assertEqual(state.final_answer, "Final grounded answer.")
        self.assertEqual(state.retrieval_attempts, 1)
        self.assertEqual(state.context_budget.selected_chunk_ids, ["c1"])
        self.assertTrue(state.response_judgment.pass_judge)

    async def test_retry_uses_query_rewriter_and_respects_attempt_cap(self) -> None:
        orchestrator = self._orchestrator(
            sufficiency_judgments=[
                SufficiencyJudgment(
                    is_sufficient=False,
                    confidence=0.4,
                    missing_aspects=["missing date"],
                    suggested_rewrite_strategy=RewriteStrategy.EXPANSION,
                ),
                SufficiencyJudgment(is_sufficient=True, confidence=0.9),
            ]
        )

        state = await orchestrator.run(
            query="NVDA revenue",
            workspace_id=1,
            db=object(),
            session_id="s1",
        )

        self.assertEqual(state.retrieval_attempts, 2)
        self.assertEqual(len(state.rewrite_history), 1)
        self.assertEqual(state.rewrite_history[0].rewritten_query, "NVDA revenue expanded")
        self.assertEqual(orchestrator.parallel_retriever.calls[1], ["NVDA revenue expanded"])

    async def test_web_search_not_called_when_disabled(self) -> None:
        web_tool = FakeWebSearchTool()
        orchestrator = self._orchestrator(
            sufficiency_judgments=[
                SufficiencyJudgment(
                    is_sufficient=False,
                    confidence=0.4,
                    missing_aspects=["fresh data"],
                    suggested_rewrite_strategy="websearch",
                )
            ],
            web_search_enabled=False,
            web_search_tool=web_tool,
        )

        await orchestrator.run(query="current NVDA price", workspace_id=1, db=object(), session_id="s1")

        self.assertEqual(web_tool.calls, 0)

    async def test_run_stream_emits_token_and_complete(self) -> None:
        orchestrator = self._orchestrator(
            sufficiency_judgments=[SufficiencyJudgment(is_sufficient=True, confidence=0.9)]
        )

        events = []
        async for event in orchestrator.run_stream(
            query="NVDA revenue 2023",
            workspace_id=1,
            db=object(),
            session_id="s1",
        ):
            events.append(event["event"])

        self.assertIn("analysis_started", events)
        self.assertIn("sources_selected", events)
        self.assertIn("generation_token", events)
        self.assertEqual(events[-1], "run_complete")


if __name__ == "__main__":
    unittest.main()
