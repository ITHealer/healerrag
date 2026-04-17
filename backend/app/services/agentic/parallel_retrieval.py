"""Parallel retrieval coordinator for Agentic RAG."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from typing import Any

from app.core.database import AsyncSessionLocal
from app.services.agentic.models import AgenticRetrievedChunk, ChunkSource, RetrievalResult

logger = logging.getLogger(__name__)

RAGServiceFactory = Callable[..., Any]


class ParallelRetrievalCoordinator:
    """Run existing RAG retrieval for multiple sub-queries and normalize output.

    This coordinator wraps current services. It does not query ChromaDB or
    LightRAG directly.
    """

    def __init__(
        self,
        *,
        service_factory: RAGServiceFactory | None = None,
        timeout_seconds: float = 10.0,
        default_top_k: int = 8,
        default_mode: str = "hybrid",
        include_images: bool = False,
        max_concurrency: int = 1,
    ) -> None:
        self._service_factory = service_factory
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._default_top_k = max(1, int(default_top_k))
        self._default_mode = default_mode or "hybrid"
        self._include_images = include_images
        self._max_concurrency = max(1, int(max_concurrency))

    @classmethod
    def from_settings(
        cls,
        *,
        service_factory: RAGServiceFactory | None = None,
        config: Any | None = None,
    ) -> "ParallelRetrievalCoordinator":
        """Build coordinator from application settings."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            service_factory=service_factory,
            timeout_seconds=config.AGENTIC_PARALLEL_RETRIEVAL_TIMEOUT,
            default_top_k=config.HEALERRAG_RERANKER_TOP_K,
            default_mode=config.HEALERRAG_DEFAULT_QUERY_MODE,
            include_images=False,
            max_concurrency=config.AGENTIC_PARALLEL_RETRIEVAL_MAX_CONCURRENCY,
        )

    async def retrieve_all(
        self,
        *,
        db: Any,
        workspace_id: int,
        sub_queries: list[str],
        entities: list[str] | None = None,
        document_ids: list[int] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        include_images: bool | None = None,
        kg_language: str | None = None,
        kg_entity_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve evidence for all sub-queries.

        One failed sub-query returns an empty result and does not fail the whole
        request.
        """

        normalized_sub_queries = self._normalize_sub_queries(sub_queries)
        if not normalized_sub_queries:
            return []

        semaphore = asyncio.Semaphore(self._max_concurrency)

        # Capture loop variables explicitly to avoid closure capture issues
        _workspace_id = workspace_id
        _kg_language = kg_language
        _kg_entity_types = kg_entity_types
        _entities = entities or []
        _document_ids = document_ids
        _metadata_filter = metadata_filter
        _top_k = top_k or self._default_top_k
        _mode = mode or self._default_mode
        _include_images = self._include_images if include_images is None else include_images

        async def _bounded_retrieve(sub_query: str) -> RetrievalResult:
            async with semaphore:
                # Each retrieval task gets its own fresh AsyncSession from the pool.
                # This prevents concurrent-use errors when the caller's session is
                # already in use (e.g. persisting the user chat message).
                async with AsyncSessionLocal() as retrieval_db:
                    service = self._build_service(
                        db=retrieval_db,
                        workspace_id=_workspace_id,
                        kg_language=_kg_language,
                        kg_entity_types=_kg_entity_types,
                    )
                    return await self._retrieve_one(
                        service=service,
                        sub_query=sub_query,
                        entities=_entities,
                        document_ids=_document_ids,
                        metadata_filter=_metadata_filter,
                        top_k=_top_k,
                        mode=_mode,
                        include_images=_include_images,
                    )

        tasks = [
            asyncio.create_task(_bounded_retrieve(sub_query))
            for sub_query in normalized_sub_queries
        ]

        return await asyncio.gather(*tasks)

    def merge_results(self, results: list[RetrievalResult]) -> list[AgenticRetrievedChunk]:
        """Deduplicate chunks and merge coverage metadata across sub-queries."""

        merged: dict[str, AgenticRetrievedChunk] = {}
        for result in results:
            for chunk in result.chunks:
                existing = merged.get(chunk.chunk_id)
                if existing is None:
                    merged[chunk.chunk_id] = chunk.model_copy(deep=True)
                    continue
                merged[chunk.chunk_id] = self._merge_chunk(existing, chunk)

        return sorted(
            merged.values(),
            key=lambda chunk: (
                -chunk.score,
                self._source_priority(chunk.source),
                chunk.chunk_id,
            ),
        )

    async def _retrieve_one(
        self,
        *,
        service: Any,
        sub_query: str,
        entities: list[str],
        document_ids: list[int] | None,
        metadata_filter: dict[str, Any] | None,
        top_k: int,
        mode: str,
        include_images: bool,
    ) -> RetrievalResult:
        try:
            result = await asyncio.wait_for(
                self._call_service(
                    service=service,
                    sub_query=sub_query,
                    document_ids=document_ids,
                    metadata_filter=metadata_filter,
                    top_k=top_k,
                    mode=mode,
                    include_images=include_images,
                ),
                timeout=self._timeout_seconds,
            )
            chunks = self._convert_result(
                raw_result=result,
                sub_query=sub_query,
                entities=entities,
            )
            return RetrievalResult(
                sub_query=sub_query,
                chunks=chunks,
                reranked=hasattr(service, "query_deep") and mode != "vector_only",
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}".strip()
            logger.warning("Agentic retrieval failed for sub-query %r: %s", sub_query, error_message)
            return RetrievalResult(
                sub_query=sub_query,
                chunks=[],
                reranked=False,
                error_message=error_message,
            )

    async def _call_service(
        self,
        *,
        service: Any,
        sub_query: str,
        document_ids: list[int] | None,
        metadata_filter: dict[str, Any] | None,
        top_k: int,
        mode: str,
        include_images: bool,
    ) -> Any:
        if hasattr(service, "query_deep") and mode != "vector_only":
            return await service.query_deep(
                question=sub_query,
                top_k=top_k,
                document_ids=document_ids,
                mode=mode,
                include_images=include_images,
                metadata_filter=metadata_filter,
            )

        def _query_sync() -> Any:
            try:
                return service.query(
                    question=sub_query,
                    top_k=top_k,
                    document_ids=document_ids,
                    metadata_filter=metadata_filter,
                )
            except TypeError:
                return service.query(
                    question=sub_query,
                    top_k=top_k,
                    document_ids=document_ids,
                )

        return await asyncio.to_thread(_query_sync)

    def _build_service(
        self,
        *,
        db: Any,
        workspace_id: int,
        kg_language: str | None,
        kg_entity_types: list[str] | None,
    ) -> Any:
        factory = self._service_factory
        if factory is None:
            from app.services.rag_service import get_rag_service

            factory = get_rag_service

        return factory(
            db,
            workspace_id,
            kg_language=kg_language,
            kg_entity_types=kg_entity_types,
        )

    def _convert_result(
        self,
        *,
        raw_result: Any,
        sub_query: str,
        entities: list[str],
    ) -> list[AgenticRetrievedChunk]:
        chunks: list[AgenticRetrievedChunk] = []

        if getattr(raw_result, "knowledge_graph_summary", ""):
            chunks.append(
                self._kg_summary_to_chunk(
                    raw_result=raw_result,
                    sub_query=sub_query,
                    entities=entities,
                )
            )

        raw_chunks = list(getattr(raw_result, "chunks", []) or [])
        citations = list(getattr(raw_result, "citations", []) or [])
        image_refs = list(getattr(raw_result, "image_refs", []) or [])
        table_refs = list(getattr(raw_result, "table_refs", []) or [])
        total = max(1, len(raw_chunks))

        for index, raw_chunk in enumerate(raw_chunks):
            citation = citations[index] if index < len(citations) else None
            chunks.append(
                self._chunk_to_agentic(
                    raw_chunk=raw_chunk,
                    citation=citation,
                    sub_query=sub_query,
                    entities=entities,
                    rank=index,
                    total=total,
                    image_refs=image_refs,
                    table_refs=table_refs,
                )
            )

        return chunks

    def _chunk_to_agentic(
        self,
        *,
        raw_chunk: Any,
        citation: Any,
        sub_query: str,
        entities: list[str],
        rank: int,
        total: int,
        image_refs: list[Any],
        table_refs: list[Any],
    ) -> AgenticRetrievedChunk:
        metadata = self._metadata_from_chunk(raw_chunk)
        if citation is not None:
            metadata["citation"] = self._citation_to_dict(citation)

        metadata["covered_sub_query"] = sub_query
        metadata["covered_sub_queries"] = [sub_query]
        metadata["covered_entities"] = entities
        metadata.setdefault("image_refs", self._safe_list(getattr(raw_chunk, "image_refs", [])))
        metadata.setdefault("table_refs", self._safe_list(getattr(raw_chunk, "table_refs", [])))
        metadata["result_image_refs"] = [self._image_to_dict(image) for image in image_refs]
        metadata["result_table_refs"] = [self._table_to_dict(table) for table in table_refs]

        return AgenticRetrievedChunk(
            chunk_id=self._chunk_id(raw_chunk),
            content=str(getattr(raw_chunk, "content", "")),
            score=self._normalized_score(raw_chunk, rank, total),
            source=ChunkSource.VECTOR,
            metadata=metadata,
        )

    @staticmethod
    def _kg_summary_to_chunk(
        *,
        raw_result: Any,
        sub_query: str,
        entities: list[str],
    ) -> AgenticRetrievedChunk:
        summary = str(getattr(raw_result, "knowledge_graph_summary", "")).strip()
        query_hash = hashlib.sha256(f"{sub_query}|{summary}".encode("utf-8")).hexdigest()[:16]
        return AgenticRetrievedChunk(
            chunk_id=f"kg_{query_hash}",
            content=summary,
            score=0.95,
            source=ChunkSource.KG,
            metadata={
                "covered_sub_query": sub_query,
                "covered_sub_queries": [sub_query],
                "covered_entities": entities,
                "source_type": "kg",
                "mode": getattr(raw_result, "mode", "hybrid"),
            },
        )

    @staticmethod
    def _metadata_from_chunk(raw_chunk: Any) -> dict[str, Any]:
        raw_metadata = getattr(raw_chunk, "metadata", None)
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            metadata = {}

        heading_path = getattr(raw_chunk, "heading_path", metadata.get("heading_path", []))
        if isinstance(heading_path, str):
            heading_path_list = [part.strip() for part in heading_path.split(">") if part.strip()]
        elif isinstance(heading_path, list):
            heading_path_list = [str(part) for part in heading_path]
        else:
            heading_path_list = []

        metadata.update(
            {
                "document_id": getattr(raw_chunk, "document_id", metadata.get("document_id", 0)),
                "page_no": getattr(raw_chunk, "page_no", metadata.get("page_no", 0)),
                "heading_path": heading_path_list,
                "source_file": getattr(
                    raw_chunk,
                    "source_file",
                    metadata.get("source_file") or metadata.get("source", ""),
                ),
                "chunk_index": getattr(raw_chunk, "chunk_index", metadata.get("chunk_index", 0)),
                "has_table": getattr(raw_chunk, "has_table", metadata.get("has_table", False)),
                "has_code": getattr(raw_chunk, "has_code", metadata.get("has_code", False)),
                "source_type": "vector",
            }
        )
        image_refs = getattr(raw_chunk, "image_refs", None)
        if image_refs is None:
            image_refs = str(metadata.get("image_ids", "")).split("|") if metadata.get("image_ids") else []
        table_refs = getattr(raw_chunk, "table_refs", None)
        if table_refs is None:
            table_refs = str(metadata.get("table_ids", "")).split("|") if metadata.get("table_ids") else []
        metadata["image_refs"] = [ref for ref in image_refs if ref]
        metadata["table_refs"] = [ref for ref in table_refs if ref]
        return metadata

    @staticmethod
    def _chunk_id(raw_chunk: Any) -> str:
        explicit = getattr(raw_chunk, "chunk_id", "")
        if explicit:
            return str(explicit)

        document_id = getattr(raw_chunk, "document_id", 0)
        chunk_index = getattr(raw_chunk, "chunk_index", None)
        if chunk_index is None:
            metadata = getattr(raw_chunk, "metadata", {}) or {}
            chunk_index = metadata.get("chunk_index", 0)
            document_id = metadata.get("document_id", document_id)
        return f"doc_{document_id}_chunk_{chunk_index}"

    @staticmethod
    def _normalized_score(raw_chunk: Any, rank: int, total: int) -> float:
        raw_score = getattr(raw_chunk, "score", None)
        if isinstance(raw_score, int | float):
            if raw_score < 0:
                return 0.0
            return 1.0 / (1.0 + float(raw_score))
        return 1.0 - (rank / max(total, 1)) * 0.5

    @staticmethod
    def _citation_to_dict(citation: Any) -> dict[str, Any]:
        return {
            "source_file": getattr(citation, "source_file", ""),
            "document_id": getattr(citation, "document_id", 0),
            "page_no": getattr(citation, "page_no", 0),
            "heading_path": list(getattr(citation, "heading_path", []) or []),
            "formatted": citation.format() if hasattr(citation, "format") else "",
        }

    @staticmethod
    def _image_to_dict(image: Any) -> dict[str, Any]:
        return {
            "image_id": getattr(image, "image_id", ""),
            "document_id": getattr(image, "document_id", 0),
            "page_no": getattr(image, "page_no", 0),
            "file_path": getattr(image, "file_path", ""),
            "caption": getattr(image, "caption", ""),
            "width": getattr(image, "width", 0),
            "height": getattr(image, "height", 0),
            "mime_type": getattr(image, "mime_type", ""),
        }

    @staticmethod
    def _table_to_dict(table: Any) -> dict[str, Any]:
        return {
            "table_id": getattr(table, "table_id", ""),
            "document_id": getattr(table, "document_id", 0),
            "page_no": getattr(table, "page_no", 0),
            "caption": getattr(table, "caption", ""),
            "num_rows": getattr(table, "num_rows", 0),
            "num_cols": getattr(table, "num_cols", 0),
        }

    @staticmethod
    def _merge_chunk(
        existing: AgenticRetrievedChunk,
        incoming: AgenticRetrievedChunk,
    ) -> AgenticRetrievedChunk:
        winner = existing if existing.score >= incoming.score else incoming
        loser = incoming if winner is existing else existing
        metadata = dict(winner.metadata)
        metadata["covered_sub_queries"] = sorted(
            set(existing.metadata.get("covered_sub_queries", []))
            | set(incoming.metadata.get("covered_sub_queries", []))
        )
        metadata["covered_entities"] = sorted(
            set(existing.metadata.get("covered_entities", []))
            | set(incoming.metadata.get("covered_entities", []))
        )
        metadata["merged_scores"] = {
            "kept": winner.score,
            "merged": loser.score,
        }
        return winner.model_copy(update={"metadata": metadata})

    @staticmethod
    def _normalize_sub_queries(sub_queries: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for sub_query in sub_queries:
            value = sub_query.strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        return normalized

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _source_priority(source: ChunkSource) -> int:
        priorities = {
            ChunkSource.KG: 0,
            ChunkSource.VECTOR: 1,
            ChunkSource.WEB: 2,
        }
        return priorities.get(source, 99)


__all__ = ["ParallelRetrievalCoordinator"]
