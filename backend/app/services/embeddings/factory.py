"""Factory helpers for shared embedding service instances."""

from __future__ import annotations

from typing import Sequence

from app.services.embeddings.embedder import EmbeddingService, get_embedding_service


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts)


__all__ = ["EmbeddingService", "get_embedding_service", "embed_text", "embed_texts"]
