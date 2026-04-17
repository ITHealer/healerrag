import unittest

from app.services.agentic.context_budget_manager import ContextBudgetManager
from app.services.agentic.models import AgenticRetrievedChunk, ChunkSource


def make_chunk(
    chunk_id: str,
    sub_query: str,
    score: float,
    content: str = "short factual evidence",
    source: ChunkSource = ChunkSource.VECTOR,
) -> AgenticRetrievedChunk:
    return AgenticRetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        score=score,
        source=source,
        metadata={
            "covered_sub_query": sub_query,
            "covered_sub_queries": [sub_query],
        },
    )


class ContextBudgetManagerTests(unittest.TestCase):
    def test_single_sub_query_caps_chunks_by_score(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=5000,
            max_final_chunks=8,
            max_chunks_per_subquery=2,
        )
        chunks = [
            make_chunk(f"chunk_{index}", "login", score=1.0 - index / 100)
            for index in range(20)
        ]

        decision = manager.select(chunks=chunks, sub_queries=["login"])

        self.assertEqual(decision.selected_chunk_ids, ["chunk_0", "chunk_1"])
        self.assertEqual(len(decision.dropped_chunk_ids), 18)
        self.assertIn("selected 2/20 chunks", decision.reasoning)

    def test_preserves_one_chunk_per_sub_query_when_budget_allows(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=5000,
            max_final_chunks=4,
            max_chunks_per_subquery=2,
        )
        chunks = [
            make_chunk("login_weak", "login", score=0.2),
            make_chunk("reset_strong", "reset password", score=0.95),
            make_chunk("forgot_strong", "forgot password", score=0.9),
            make_chunk("login_strong", "login", score=0.8),
            make_chunk("reset_extra", "reset password", score=0.7),
        ]

        decision = manager.select(
            chunks=chunks,
            sub_queries=["login", "reset password", "forgot password"],
        )

        self.assertIn("login_strong", decision.selected_chunk_ids)
        self.assertIn("reset_strong", decision.selected_chunk_ids)
        self.assertIn("forgot_strong", decision.selected_chunk_ids)
        self.assertLessEqual(len(decision.selected_chunk_ids), 4)

    def test_enforces_token_budget_deterministically(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=30,
            max_final_chunks=5,
            max_chunks_per_subquery=2,
        )
        long_content = " ".join(["evidence"] * 80)
        chunks = [
            make_chunk("a_low_long", "a", score=0.2, content=long_content),
            make_chunk("a_high_short", "a", score=0.9, content="compact a evidence"),
            make_chunk("b_high_short", "b", score=0.8, content="compact b evidence"),
        ]

        first = manager.select(chunks=chunks, sub_queries=["a", "b"])
        second = manager.select(chunks=chunks, sub_queries=["a", "b"])

        self.assertEqual(first.selected_chunk_ids, second.selected_chunk_ids)
        self.assertIn("a_high_short", first.selected_chunk_ids)
        self.assertIn("b_high_short", first.selected_chunk_ids)
        self.assertNotIn("a_low_long", first.selected_chunk_ids)
        self.assertIn("budget_drops=1", first.reasoning)

    def test_tie_breaks_by_source_priority_then_chunk_id(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=5000,
            max_final_chunks=2,
            max_chunks_per_subquery=2,
        )
        chunks = [
            make_chunk("vector_b", "login", score=0.7, source=ChunkSource.VECTOR),
            make_chunk("kg_a", "login", score=0.7, source=ChunkSource.KG),
            make_chunk("web_c", "login", score=0.7, source=ChunkSource.WEB),
        ]

        decision = manager.select(chunks=chunks, sub_queries=["login"])

        self.assertEqual(decision.selected_chunk_ids, ["kg_a", "vector_b"])

    def test_dedupes_duplicate_chunk_ids_by_best_score(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=5000,
            max_final_chunks=3,
            max_chunks_per_subquery=2,
        )
        chunks = [
            make_chunk("same", "login", score=0.1),
            make_chunk("same", "login", score=0.9),
            make_chunk("other", "login", score=0.8),
        ]

        decision = manager.select(chunks=chunks, sub_queries=["login"])

        self.assertEqual(decision.selected_chunk_ids, ["same", "other"])
        self.assertEqual(decision.dropped_chunk_ids, [])

    def test_chunks_without_matching_sub_query_can_fill_remaining_slots(self) -> None:
        manager = ContextBudgetManager(
            max_final_context_tokens=5000,
            max_final_chunks=2,
            max_chunks_per_subquery=2,
        )
        chunks = [
            make_chunk("login", "login", score=0.6),
            make_chunk("related", "related topic", score=0.95),
        ]

        decision = manager.select(chunks=chunks, sub_queries=["login"])

        self.assertEqual(decision.selected_chunk_ids, ["login", "related"])


if __name__ == "__main__":
    unittest.main()
