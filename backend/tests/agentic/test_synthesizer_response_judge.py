import json
import unittest

from app.services.agentic.hierarchical_synthesizer import HierarchicalSynthesizer
from app.services.agentic.models import (
    AgenticRetrievedChunk,
    ChunkSource,
    ExecutionItem,
    ExecutionPlan,
)
from app.services.agentic.response_judge import ResponseJudge


class FakeLLMProvider:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def acomplete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _chunk(
    chunk_id: str,
    content: str,
    *,
    sub_query: str,
    score: float = 0.8,
) -> AgenticRetrievedChunk:
    return AgenticRetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source=ChunkSource.VECTOR,
        metadata={"covered_sub_query": sub_query, "citation": {"chunk_id": chunk_id}},
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


class HierarchicalSynthesizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_summarize_preserves_supporting_chunk_ids(self) -> None:
        provider = FakeLLMProvider("Revenue increased and margin improved [c1][c2].")
        synthesizer = HierarchicalSynthesizer(llm_provider=provider)
        chunks = [
            _chunk("c1", "NVDA revenue increased in 2023.", sub_query="NVDA revenue 2023"),
            _chunk("c2", "NVDA margin improved in 2023.", sub_query="NVDA revenue 2023"),
        ]

        summaries = await synthesizer.summarize(chunks=chunks, sub_queries=["NVDA revenue 2023"])

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].supporting_chunk_ids, ["c1", "c2"])
        self.assertIn("Revenue increased", summaries[0].summary)
        self.assertEqual(len(provider.calls), 1)

    async def test_summary_fallback_keeps_citation_mapping(self) -> None:
        synthesizer = HierarchicalSynthesizer(llm_provider=FakeLLMProvider(""), max_excerpt_chars=80)
        chunks = [
            _chunk("c1", "NVDA revenue evidence text.", sub_query="NVDA revenue 2023"),
            _chunk("c2", "NVDA margin evidence text.", sub_query="NVDA margin 2023"),
        ]

        summaries = await synthesizer.summarize(
            chunks=chunks,
            sub_queries=["NVDA revenue 2023", "NVDA margin 2023"],
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].supporting_chunk_ids, ["c1"])
        self.assertEqual(summaries[1].supporting_chunk_ids, ["c2"])
        self.assertIn("c1:", summaries[0].summary)

    async def test_assemble_includes_summaries_and_critical_evidence(self) -> None:
        synthesizer = HierarchicalSynthesizer(use_llm=False, max_excerpt_chars=120)
        chunks = [
            _chunk(
                "c1",
                "NVDA revenue increased in 2023. " * 20,
                sub_query="NVDA revenue 2023",
            ),
            _chunk(
                "c2",
                "NVDA gross margin improved in 2023. " * 20,
                sub_query="NVDA margin 2023",
            ),
        ]

        summaries = await synthesizer.summarize(
            chunks=chunks,
            sub_queries=["NVDA revenue 2023", "NVDA margin 2023"],
        )
        assembled = synthesizer.assemble(
            original_query="Analyze NVDA revenue and margin",
            summaries=summaries,
            chunks=chunks,
        )

        raw_context = "\n".join(chunk.content for chunk in chunks)
        self.assertLess(len(assembled), len(raw_context))
        self.assertIn("Sub-query summaries:", assembled)
        self.assertIn("Critical evidence:", assembled)
        self.assertIn("Supported by: c1", assembled)
        self.assertIn("chunk_id=c2", assembled)


class ResponseJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_llm_json_passes_above_thresholds(self) -> None:
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "pass_judge": True,
                    "faithfulness_score": 0.95,
                    "completeness_score": 0.9,
                    "issues": [],
                    "reasoning": "Grounded and complete.",
                }
            )
        )
        judge = ResponseJudge(llm_provider=provider)

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA revenue and margin are covered.",
            chunks=[_chunk("c1", "NVDA revenue and margin are covered.", sub_query="NVDA revenue 2023")],
            execution_plan=_plan(),
        )

        self.assertTrue(result.pass_judge)
        self.assertEqual(len(provider.calls), 1)

    async def test_llm_scores_below_threshold_fail(self) -> None:
        provider = FakeLLMProvider(
            json.dumps(
                {
                    "pass_judge": True,
                    "faithfulness_score": 0.5,
                    "completeness_score": 0.4,
                    "issues": [],
                    "reasoning": "Weak answer.",
                }
            )
        )
        judge = ResponseJudge(llm_provider=provider, faithfulness_threshold=0.7, completeness_threshold=0.7)

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA did something.",
            chunks=[_chunk("c1", "NVDA revenue and margin evidence.", sub_query="NVDA revenue 2023")],
            execution_plan=_plan(),
        )

        self.assertFalse(result.pass_judge)
        self.assertIn("Faithfulness score is below threshold.", result.issues)
        self.assertIn("Completeness score is below threshold.", result.issues)

    async def test_rule_based_good_grounded_answer_passes(self) -> None:
        judge = ResponseJudge(use_llm=False, faithfulness_threshold=0.55, completeness_threshold=0.7)
        chunks = [
            _chunk(
                "c1",
                "NVDA revenue 2023 increased and NVDA margin 2023 improved.",
                sub_query="NVDA revenue 2023",
            )
        ]

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA revenue 2023 increased. NVDA margin 2023 improved.",
            chunks=chunks,
            execution_plan=_plan(),
        )

        self.assertTrue(result.pass_judge)
        self.assertGreaterEqual(result.faithfulness_score, 0.55)
        self.assertEqual(result.completeness_score, 1.0)

    async def test_rule_based_unsupported_answer_fails(self) -> None:
        judge = ResponseJudge(use_llm=False, faithfulness_threshold=0.7, completeness_threshold=0.7)
        chunks = [_chunk("c1", "NVDA revenue 2023 increased.", sub_query="NVDA revenue 2023")]

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA revenue 2023 increased. NVDA acquired a robotics company.",
            chunks=chunks,
            execution_plan=_plan(),
        )

        self.assertFalse(result.pass_judge)
        self.assertTrue(any("not well supported" in issue for issue in result.issues))

    async def test_rule_based_incomplete_batch_answer_fails(self) -> None:
        judge = ResponseJudge(use_llm=False, faithfulness_threshold=0.5, completeness_threshold=0.9)
        chunks = [
            _chunk(
                "c1",
                "NVDA revenue 2023 increased and NVDA margin 2023 improved.",
                sub_query="NVDA revenue 2023",
            )
        ]

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA revenue 2023 increased.",
            chunks=chunks,
            execution_plan=_plan(),
        )

        self.assertFalse(result.pass_judge)
        self.assertIn("Missing planned item: NVDA margin 2023", result.issues)

    async def test_invalid_llm_json_uses_rule_based_fallback(self) -> None:
        judge = ResponseJudge(
            llm_provider=FakeLLMProvider("not json"),
            faithfulness_threshold=0.55,
            completeness_threshold=0.7,
        )
        chunks = [
            _chunk(
                "c1",
                "NVDA revenue 2023 increased and NVDA margin 2023 improved.",
                sub_query="NVDA revenue 2023",
            )
        ]

        result = await judge.judge(
            original_query="Analyze NVDA revenue and margin",
            generated_answer="NVDA revenue 2023 increased. NVDA margin 2023 improved.",
            chunks=chunks,
            execution_plan=_plan(),
        )

        self.assertTrue(result.pass_judge)


if __name__ == "__main__":
    unittest.main()
