import unittest

from app.services.agentic.models import ChunkSource, RetrievalResult
from app.services.agentic.parallel_retrieval import ParallelRetrievalCoordinator
from app.services.models.parsed_document import (
    Citation,
    DeepRetrievalResult,
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
)
from app.services.rag_service import RAGQueryResult, RetrievedChunk


class FakeDeepService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def query_deep(
        self,
        *,
        question: str,
        top_k: int,
        document_ids: list[int] | None,
        mode: str,
        include_images: bool,
        metadata_filter: dict | None,
    ) -> DeepRetrievalResult:
        self.calls.append(question)
        if question == "bad":
            raise RuntimeError("simulated retrieval failure")

        chunk = EnrichedChunk(
            content=f"content for {question}",
            chunk_index=0,
            source_file="manual.pdf",
            document_id=10,
            page_no=2,
            heading_path=["Login"],
            image_refs=["img_1"],
            table_refs=["tbl_1"],
            has_table=True,
        )
        return DeepRetrievalResult(
            chunks=[chunk],
            citations=[
                Citation(
                    source_file="manual.pdf",
                    document_id=10,
                    page_no=2,
                    heading_path=["Login"],
                )
            ],
            context="context",
            query=question,
            mode=mode,
            knowledge_graph_summary=f"KG facts for {question}",
            image_refs=[
                ExtractedImage(
                    image_id="img_1",
                    document_id=10,
                    page_no=2,
                    file_path="image.png",
                    caption="login screen",
                )
            ],
            table_refs=[
                ExtractedTable(
                    table_id="tbl_1",
                    document_id=10,
                    page_no=2,
                    caption="login table",
                    num_rows=2,
                    num_cols=3,
                )
            ],
        )


class FakeLegacyService:
    def query(
        self,
        *,
        question: str,
        top_k: int,
        document_ids: list[int] | None,
        metadata_filter: dict | None = None,
    ) -> RAGQueryResult:
        return RAGQueryResult(
            chunks=[
                RetrievedChunk(
                    content=f"legacy content for {question}",
                    metadata={
                        "document_id": 7,
                        "chunk_index": 3,
                        "source": "legacy.pdf",
                        "page_no": 5,
                        "heading_path": "Legacy > Login",
                        "image_ids": "img_7|",
                        "table_ids": "tbl_7",
                    },
                    score=0.25,
                    chunk_id="legacy_chunk",
                )
            ],
            context="legacy context",
            query=question,
        )


class ParallelRetrievalCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_all_uses_deep_query_and_isolates_failures(self) -> None:
        service = FakeDeepService()
        coordinator = ParallelRetrievalCoordinator(
            service_factory=lambda *args, **kwargs: service,
            timeout_seconds=2,
            default_top_k=3,
        )

        results = await coordinator.retrieve_all(
            db=object(),
            workspace_id=1,
            sub_queries=["login", "bad", "login"],
            entities=["BESTmed"],
            document_ids=[10],
            metadata_filter={"kind": "guide"},
        )

        self.assertEqual([result.sub_query for result in results], ["login", "bad"])
        self.assertEqual(service.calls, ["login", "bad"])
        self.assertEqual(len(results[0].chunks), 2)
        self.assertEqual(results[0].chunks[0].source, ChunkSource.KG)
        vector_chunk = results[0].chunks[1]
        self.assertEqual(vector_chunk.chunk_id, "doc_10_chunk_0")
        self.assertEqual(vector_chunk.metadata["citation"]["formatted"], "manual.pdf | p.2 | Login")
        self.assertEqual(vector_chunk.metadata["image_refs"], ["img_1"])
        self.assertEqual(vector_chunk.metadata["table_refs"], ["tbl_1"])
        self.assertEqual(vector_chunk.metadata["result_image_refs"][0]["image_id"], "img_1")
        self.assertEqual(vector_chunk.metadata["covered_entities"], ["BESTmed"])
        self.assertEqual(results[1].chunks, [])

    async def test_retrieve_all_falls_back_to_legacy_query(self) -> None:
        coordinator = ParallelRetrievalCoordinator(
            service_factory=lambda *args, **kwargs: FakeLegacyService(),
            default_mode="vector_only",
        )

        results = await coordinator.retrieve_all(
            db=object(),
            workspace_id=1,
            sub_queries=["legacy login"],
        )

        self.assertEqual(len(results), 1)
        chunk = results[0].chunks[0]
        self.assertEqual(chunk.chunk_id, "legacy_chunk")
        self.assertEqual(chunk.content, "legacy content for legacy login")
        self.assertEqual(chunk.metadata["heading_path"], ["Legacy", "Login"])
        self.assertEqual(chunk.metadata["image_refs"], ["img_7"])
        self.assertEqual(chunk.metadata["table_refs"], ["tbl_7"])
        self.assertGreater(chunk.score, 0.7)

    def test_merge_results_dedupes_and_merges_coverage(self) -> None:
        coordinator = ParallelRetrievalCoordinator()
        first = FakeLegacyService().query(
            question="q1",
            top_k=1,
            document_ids=None,
        )
        second = FakeLegacyService().query(
            question="q2",
            top_k=1,
            document_ids=None,
        )

        chunks_q1 = coordinator._convert_result(raw_result=first, sub_query="q1", entities=["A"])
        chunks_q2 = coordinator._convert_result(raw_result=second, sub_query="q2", entities=["B"])
        chunks_q2[0].score = 0.99
        results = [
            RetrievalResult(sub_query="q1", chunks=chunks_q1, reranked=False),
            RetrievalResult(sub_query="q2", chunks=chunks_q2, reranked=False),
        ]

        merged = coordinator.merge_results(results)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].score, 0.99)
        self.assertEqual(merged[0].metadata["covered_sub_queries"], ["q1", "q2"])
        self.assertEqual(merged[0].metadata["covered_entities"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
