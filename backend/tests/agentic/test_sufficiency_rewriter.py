import json
import unittest

from app.services.agentic.models import (
    AgenticRetrievedChunk,
    ChunkSource,
    ExecutionItem,
    ExecutionPlan,
    QueryAnalysisResult,
    QueryComplexity,
    RewriteStrategy,
)
from app.services.agentic.query_rewriter import QueryRewriter
from app.services.agentic.sufficiency_judge import SufficiencyJudge


class FakeLLMProvider:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def acomplete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        complexity=QueryComplexity.MULTI_HOP,
        sub_queries=["NVDA revenue 2023", "NVDA margin 2023"],
        entities=["NVDA"],
        temporal_range="2023",
        language="en",
    )


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        can_fully_answer_now=True,
        total_items=2,
        items=[
            ExecutionItem(
                item_id="item_1",
                description="NVDA revenue 2023",
                priority=1,
                estimated_output_tokens=300,
                related_sub_queries=["NVDA revenue 2023"],
            ),
            ExecutionItem(
                item_id="item_2",
                description="NVDA margin 2023",
                priority=2,
                estimated_output_tokens=300,
                related_sub_queries=["NVDA margin 2023"],
            ),
        ],
        batch_now=["item_1", "item_2"],
        batch_later=[],
    )


def _chunk(chunk_id: str, content: str, score: float = 0.8) -> AgenticRetrievedChunk:
    return AgenticRetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source=ChunkSource.VECTOR,
        metadata={"covered_sub_query": "NVDA revenue 2023"},
    )


class SufficiencyJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_chunks_are_insufficient(self) -> None:
        judge = SufficiencyJudge(use_llm=False)

        result = await judge.judge(
            original_query="Analyze NVDA revenue",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=[],
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.suggested_rewrite_strategy, RewriteStrategy.EXPANSION)
        self.assertIn("No retrieved evidence", result.missing_aspects[0])

    async def test_valid_llm_json_above_threshold_passes(self) -> None:
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "is_sufficient": True,
                    "confidence": 0.9,
                    "missing_aspects": [],
                    "covered_aspects": ["revenue", "margin"],
                    "suggested_rewrite_strategy": None,
                    "reasoning": "Evidence covers both items.",
                }
            )
        )
        judge = SufficiencyJudge(llm_provider=provider, sufficiency_threshold=0.7)

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=[_chunk("c1", "NVDA revenue 2023 and margin 2023 are both discussed.")],
        )

        self.assertTrue(result.is_sufficient)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(len(provider.calls), 1)

    async def test_llm_confidence_below_threshold_becomes_insufficient(self) -> None:
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "is_sufficient": True,
                    "confidence": 0.4,
                    "missing_aspects": [],
                    "covered_aspects": ["some evidence"],
                    "suggested_rewrite_strategy": None,
                    "reasoning": "Low confidence.",
                }
            )
        )
        judge = SufficiencyJudge(llm_provider=provider, sufficiency_threshold=0.7)

        result = await judge.judge(
            original_query="Analyze NVDA revenue",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=[_chunk("c1", "NVDA revenue 2023")],
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.suggested_rewrite_strategy, RewriteStrategy.EXPANSION)
        self.assertIn("threshold", result.missing_aspects[0])

    async def test_invalid_llm_json_degrades_to_sufficient_low_confidence(self) -> None:
        judge = SufficiencyJudge(llm_provider=FakeLLMProvider("not json"))

        result = await judge.judge(
            original_query="Analyze NVDA revenue",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=[_chunk("c1", "some retrieved evidence")],
        )

        self.assertTrue(result.is_sufficient)
        self.assertEqual(result.confidence, 0.35)
        self.assertEqual(result.missing_aspects, [])

    async def test_rule_based_detects_missing_aspects(self) -> None:
        judge = SufficiencyJudge(use_llm=False, sufficiency_threshold=0.7)

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=[_chunk("c1", "NVDA revenue 2023 increased.")],
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.suggested_rewrite_strategy, RewriteStrategy.EXPANSION)
        self.assertIn("NVDA margin 2023", result.missing_aspects)

    async def test_rule_based_detects_overloaded_context(self) -> None:
        chunks = [_chunk(f"c{i}", f"unfocused evidence {i}") for i in range(7)]
        judge = SufficiencyJudge(use_llm=False, max_evidence_items=2, overload_chunk_count=6)

        result = await judge.judge(
            original_query="Analyze NVDA revenue",
            analysis=_analysis(),
            execution_plan=_plan(),
            chunks=chunks,
        )

        self.assertFalse(result.is_sufficient)
        self.assertEqual(result.suggested_rewrite_strategy, RewriteStrategy.STEP_BACK)
        self.assertIn("overloaded", result.missing_aspects[0])


class QueryRewriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_llm_json_is_normalized_to_requested_contract(self) -> None:
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "original_query": "old",
                    "rewritten_query": "  NVDA 2023 revenue annual report evidence  ",
                    "strategy": "hyde",
                    "iteration": 99,
                }
            )
        )
        rewriter = QueryRewriter(llm_provider=provider)

        result = await rewriter.rewrite(
            original_query="NVDA revenue",
            missing_aspects=["2023 annual report"],
            strategy=RewriteStrategy.EXPANSION,
            iteration=2,
        )

        self.assertEqual(result.original_query, "NVDA revenue")
        self.assertEqual(result.rewritten_query, "NVDA 2023 revenue annual report evidence")
        self.assertEqual(result.strategy, RewriteStrategy.EXPANSION)
        self.assertEqual(result.iteration, 2)

    async def test_expansion_fallback_includes_missing_aspects(self) -> None:
        rewriter = QueryRewriter(use_llm=False)

        result = await rewriter.rewrite(
            original_query="NVDA revenue",
            missing_aspects=["2023", "annual report"],
            strategy=RewriteStrategy.EXPANSION,
            iteration=1,
        )

        self.assertEqual(result.strategy, RewriteStrategy.EXPANSION)
        self.assertEqual(result.iteration, 1)
        self.assertIn("2023", result.rewritten_query)
        self.assertIn("annual report", result.rewritten_query)

    async def test_step_back_fallback_is_broader(self) -> None:
        rewriter = QueryRewriter(use_llm=False)

        result = await rewriter.rewrite(
            original_query="NVDA gross margin 2023",
            missing_aspects=[],
            strategy=RewriteStrategy.STEP_BACK,
            iteration=3,
        )

        self.assertEqual(result.strategy, RewriteStrategy.STEP_BACK)
        self.assertEqual(result.iteration, 3)
        self.assertIn("Broader background", result.rewritten_query)

    async def test_hyde_fallback_is_answer_like_for_retrieval(self) -> None:
        rewriter = QueryRewriter(use_llm=False)

        result = await rewriter.rewrite(
            original_query="NVDA free cash flow",
            missing_aspects=["FY2023"],
            strategy=RewriteStrategy.HYDE,
            iteration=1,
        )

        self.assertEqual(result.strategy, RewriteStrategy.HYDE)
        self.assertIn("Hypothetical relevant passage", result.rewritten_query)
        self.assertIn("FY2023", result.rewritten_query)

    async def test_invalid_llm_json_uses_requested_strategy_fallback(self) -> None:
        rewriter = QueryRewriter(llm_provider=FakeLLMProvider("not json"))

        result = await rewriter.rewrite(
            original_query="NVDA revenue",
            missing_aspects=["10-K"],
            strategy=RewriteStrategy.EXPANSION,
            iteration=1,
        )

        self.assertEqual(result.strategy, RewriteStrategy.EXPANSION)
        self.assertIn("10-K", result.rewritten_query)


if __name__ == "__main__":
    unittest.main()
