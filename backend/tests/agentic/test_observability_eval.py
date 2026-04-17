from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.agentic.models import (
    AgenticRAGState,
    AgenticRetrievedChunk,
    ChunkSource,
    ContextBudgetDecision,
    QueryAnalysisResult,
    QueryComplexity,
)
from app.services.agentic.observability import agentic_log_extra, state_observability_metadata
from scripts.eval_agentic_rag import DEFAULT_DATASET, EvalConfig, evaluate_dataset


class AgenticObservabilityTests(unittest.TestCase):
    def test_state_metadata_exposes_debug_counts_without_content(self) -> None:
        state = AgenticRAGState(
            original_query="secret user query",
            workspace_id="1",
            analysis=QueryAnalysisResult(
                complexity=QueryComplexity.SINGLE_HOP,
                sub_queries=["secret sub query"],
                language="en",
            ),
            merged_chunks=[
                AgenticRetrievedChunk(
                    chunk_id="chunk_1",
                    content="secret document content",
                    score=0.9,
                    source=ChunkSource.VECTOR,
                )
            ],
            context_budget=ContextBudgetDecision(
                max_final_context_tokens=1000,
                max_final_chunks=3,
                max_chunks_per_subquery=1,
                selected_chunk_ids=["chunk_1"],
                dropped_chunk_ids=["chunk_2"],
            ),
            retrieval_attempts=1,
        )

        metadata = state_observability_metadata(state)
        self.assertEqual(metadata["complexity"], "single_hop")
        self.assertEqual(metadata["sub_query_count"], 1)
        self.assertEqual(metadata["selected_chunk_ids"], ["chunk_1"])
        self.assertNotIn("secret user query", str(metadata))
        self.assertNotIn("secret document content", str(metadata))

    def test_log_extra_includes_run_identity_and_state_summary(self) -> None:
        state = AgenticRAGState(original_query="hello", workspace_id="7")
        extra = agentic_log_extra(
            run_id="run_1",
            workspace_id=7,
            session_id="session_1",
            state=state,
            custom_value={"nested": ChunkSource.WEB},
        )

        self.assertEqual(extra["agentic_run_id"], "run_1")
        self.assertEqual(extra["workspace_id"], "7")
        self.assertEqual(extra["session_id"], "session_1")
        self.assertEqual(extra["custom_value"], {"nested": "web"})


class AgenticEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_eval_dataset_runs_without_web_or_db(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "agentic_eval_results.json"
            report = await evaluate_dataset(
                EvalConfig(
                    dataset_path=DEFAULT_DATASET,
                    output_path=output_path,
                    max_cases=8,
                    output_budget=1800,
                )
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(report["mode"], "offline_rule_based")
            self.assertEqual(report["case_count"], 8)
            self.assertEqual(report["summary"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
