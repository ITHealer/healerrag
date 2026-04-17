import json
import unittest

from app.services.agentic.models import QueryAnalysisResult, QueryComplexity
from app.services.agentic.query_analyzer import QueryAnalyzer
from app.services.agentic.response_planner import ResponsePlanner


class FakeLLMProvider:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def acomplete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class QueryAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyzer_uses_valid_llm_json(self) -> None:
        llm_response = json.dumps(
            {
                "complexity": "multi_hop",
                "sub_queries": ["NVDA revenue 2023", "NVDA margin 2023"],
                "entities": ["NVDA"],
                "temporal_range": "2023",
                "language": "en",
                "strategy_hint": "multi_request",
                "reasoning": "Two requested metrics.",
            }
        )
        provider = FakeLLMProvider(llm_response)
        analyzer = QueryAnalyzer(llm_provider=provider)

        analysis = await analyzer.analyze("Analyze NVDA revenue and margin in 2023")

        self.assertEqual(analysis.complexity, QueryComplexity.MULTI_HOP)
        self.assertEqual(analysis.sub_queries, ["NVDA revenue 2023", "NVDA margin 2023"])
        self.assertEqual(analysis.entities, ["NVDA"])
        self.assertEqual(len(provider.calls), 1)

    async def test_analyzer_invalid_llm_json_falls_back_for_vietnamese_multi_query(self) -> None:
        analyzer = QueryAnalyzer(llm_provider=FakeLLMProvider("not json"))

        analysis = await analyzer.analyze("C\u00e1ch \u0111\u0103ng nh\u1eadp v\u00e0 reset m\u1eadt kh\u1ea9u BESTmed?")

        self.assertEqual(analysis.complexity, QueryComplexity.MULTI_HOP)
        self.assertEqual(analysis.language, "vi")
        self.assertIn("BESTmed", analysis.entities)
        self.assertGreaterEqual(len(analysis.sub_queries), 2)

    async def test_analyzer_detects_no_retrieval_greeting_without_llm(self) -> None:
        analyzer = QueryAnalyzer(use_llm=False)

        analysis = await analyzer.analyze("xin ch\u00e0o")

        self.assertEqual(analysis.complexity, QueryComplexity.NO_RETRIEVAL)
        self.assertEqual(analysis.sub_queries, [])
        self.assertEqual(analysis.strategy_hint, "no_retrieval")

    async def test_analyzer_detects_english_single_hop_without_llm(self) -> None:
        analyzer = QueryAnalyzer(use_llm=False)

        analysis = await analyzer.analyze("How do I login to BESTmed?")

        self.assertEqual(analysis.complexity, QueryComplexity.SINGLE_HOP)
        self.assertEqual(analysis.sub_queries, ["How do I login to BESTmed?"])
        self.assertEqual(analysis.language, "en")
        self.assertEqual(analysis.entities, ["BESTmed"])


class ResponsePlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_uses_valid_llm_json(self) -> None:
        analysis = QueryAnalysisResult(
            complexity=QueryComplexity.MULTI_HOP,
            sub_queries=["revenue", "margin"],
            language="en",
        )
        llm_response = json.dumps(
            {
                "can_fully_answer_now": True,
                "total_items": 2,
                "items": [
                    {
                        "item_id": "item_1",
                        "description": "Explain revenue",
                        "priority": 1,
                        "estimated_output_tokens": 300,
                        "related_sub_queries": ["revenue"],
                        "depends_on": [],
                    },
                    {
                        "item_id": "item_2",
                        "description": "Explain margin",
                        "priority": 2,
                        "estimated_output_tokens": 300,
                        "related_sub_queries": ["margin"],
                        "depends_on": [],
                    },
                ],
                "batch_now": ["item_1", "item_2"],
                "batch_later": [],
                "continuation_message": None,
                "reasoning": "Fits budget.",
            }
        )
        planner = ResponsePlanner(llm_provider=FakeLLMProvider(llm_response), max_output_tokens_per_turn=1000)

        plan = await planner.plan(query="Analyze revenue and margin", analysis=analysis)

        self.assertTrue(plan.can_fully_answer_now)
        self.assertEqual(plan.batch_now, ["item_1", "item_2"])
        self.assertEqual(plan.batch_later, [])

    async def test_planner_rebuilds_llm_batch_when_it_exceeds_budget(self) -> None:
        analysis = QueryAnalysisResult(
            complexity=QueryComplexity.MULTI_HOP,
            sub_queries=["revenue", "profit", "cash flow"],
            language="en",
        )
        llm_response = json.dumps(
            {
                "can_fully_answer_now": True,
                "total_items": 3,
                "items": [
                    {
                        "item_id": "item_1",
                        "description": "Analyze revenue",
                        "priority": 1,
                        "estimated_output_tokens": 700,
                        "related_sub_queries": ["revenue"],
                    },
                    {
                        "item_id": "item_2",
                        "description": "Analyze profit",
                        "priority": 2,
                        "estimated_output_tokens": 700,
                        "related_sub_queries": ["profit"],
                    },
                    {
                        "item_id": "item_3",
                        "description": "Analyze cash flow",
                        "priority": 3,
                        "estimated_output_tokens": 700,
                        "related_sub_queries": ["cash flow"],
                    },
                ],
                "batch_now": ["item_1", "item_2", "item_3"],
                "batch_later": [],
                "reasoning": "Incorrectly says all fit.",
            }
        )
        planner = ResponsePlanner(llm_provider=FakeLLMProvider(llm_response), max_output_tokens_per_turn=1000)

        plan = await planner.plan(query="Analyze revenue, profit, and cash flow", analysis=analysis)

        self.assertFalse(plan.can_fully_answer_now)
        self.assertEqual(plan.batch_now, ["item_1"])
        self.assertEqual(plan.batch_later, ["item_2", "item_3"])
        self.assertIn("Remaining", plan.continuation_message)

    async def test_planner_invalid_llm_json_falls_back_and_splits_vietnamese_request(self) -> None:
        analysis = QueryAnalysisResult(
            complexity=QueryComplexity.MULTI_HOP,
            sub_queries=[
                "ph\u00e2n t\u00edch doanh thu",
                "l\u1ee3i nhu\u1eadn",
                "gross margin",
                "free cash flow",
            ],
            language="vi",
        )
        planner = ResponsePlanner(
            llm_provider=FakeLLMProvider("not json"),
            max_output_tokens_per_turn=1000,
        )

        plan = await planner.plan(
            query="Ph\u00e2n t\u00edch doanh thu, l\u1ee3i nhu\u1eadn, gross margin v\u00e0 free cash flow",
            analysis=analysis,
        )

        self.assertFalse(plan.can_fully_answer_now)
        self.assertEqual(plan.total_items, 4)
        self.assertEqual(plan.batch_now, ["item_1"])
        self.assertEqual(plan.batch_later, ["item_2", "item_3", "item_4"])
        self.assertIn("ti\u1ebfp t\u1ee5c", plan.continuation_message)

    async def test_planner_simple_single_hop_fits_one_batch_without_llm(self) -> None:
        analysis = QueryAnalysisResult(
            complexity=QueryComplexity.SINGLE_HOP,
            sub_queries=["How do I login?"],
            language="en",
        )
        planner = ResponsePlanner(use_llm=False, max_output_tokens_per_turn=500)

        plan = await planner.plan(query="How do I login?", analysis=analysis)

        self.assertTrue(plan.can_fully_answer_now)
        self.assertEqual(plan.total_items, 1)
        self.assertEqual(plan.batch_now, ["item_1"])
        self.assertEqual(plan.batch_later, [])


if __name__ == "__main__":
    unittest.main()
