"""Chunking utilities for splitting and deduplicating document content."""

from app.services.chunking.chunk_dedup import deduplicate_chunks
from app.services.chunking.chunker import DocumentChunker, TextChunk, chunk_text

__all__ = [
    "DocumentChunker",
    "TextChunk",
    "chunk_text",
    "deduplicate_chunks",
]
