"""Agentic continuation session persistence model."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AgenticSessionStatus:
    """String status constants for continuation sessions."""

    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"


class AgenticSession(Base):
    """Persist unfinished Agentic RAG execution state per workspace/session."""

    __tablename__ = "agentic_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
    )
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), default=AgenticSessionStatus.ACTIVE, index=True)
    original_query: Mapped[str] = mapped_column(Text)
    state: Mapped[dict] = mapped_column(JSON)
    completed_item_ids: Mapped[list] = mapped_column(JSON, default=list)
    remaining_item_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    citations: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


__all__ = ["AgenticSession", "AgenticSessionStatus"]
