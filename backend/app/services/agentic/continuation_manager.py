"""Continuation state persistence for Agentic RAG."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agentic_session import AgenticSession, AgenticSessionStatus
from app.services.agentic.models import ContinuationState, ExecutionPlan


class ContinuationManager:
    """Save and resume unfinished Agentic RAG execution items."""

    _CONTINUATION_PATTERNS = (
        r"\bcontinue\b",
        r"\bresume\b",
        r"\bnext\b",
        r"\bgo on\b",
        r"\bkeep going\b",
        r"\bti(?:e|\u1ebf)p t(?:u|\u1ee5)c\b",
        r"\bl(?:a|\u00e0)m ti(?:e|\u1ebf)p\b",
        r"\bph(?:a|\u1ea7)n c(?:o|\u00f2)n l(?:a|\u1ea1)i\b",
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_hours: int = 24,
    ) -> None:
        self._enabled = enabled
        self._ttl_hours = max(1, int(ttl_hours))

    @classmethod
    def from_settings(cls, *, config: Any | None = None) -> "ContinuationManager":
        """Build manager from settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            enabled=config.AGENTIC_CONTINUATION_ENABLED,
            ttl_hours=config.AGENTIC_CONTINUATION_TTL_HOURS,
        )

    async def save(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
        original_query: str,
        execution_plan: ExecutionPlan,
        completed_item_ids: list[str],
        remaining_item_ids: list[str],
        evidence_chunk_ids: list[str],
        citations: list[dict[str, Any]],
        now: datetime | None = None,
    ) -> ContinuationState | None:
        """Create an active continuation state for unfinished execution items."""

        state = ContinuationState(
            session_id=session_id,
            original_query=original_query,
            execution_plan=execution_plan,
            completed_item_ids=self._dedupe(completed_item_ids),
            remaining_item_ids=self._dedupe(remaining_item_ids),
            evidence_chunk_ids=self._dedupe(evidence_chunk_ids),
            citations=citations,
        )
        return await self.save_state(db=db, workspace_id=workspace_id, state=state, now=now)

    async def save_state(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        state: ContinuationState,
        now: datetime | None = None,
    ) -> ContinuationState | None:
        """Persist a pre-built continuation state."""

        if not self._enabled:
            return None

        effective_now = now or datetime.utcnow()
        if not state.remaining_item_ids:
            return None

        expires_at = effective_now + timedelta(hours=self._ttl_hours)
        await self._expire_active_records(
            db=db,
            workspace_id=workspace_id,
            session_id=state.session_id,
            now=effective_now,
        )

        state_json = state.model_dump(mode="json")
        record = AgenticSession(
            workspace_id=workspace_id,
            session_id=state.session_id,
            status=AgenticSessionStatus.ACTIVE,
            original_query=state.original_query,
            state=state_json,
            completed_item_ids=state.completed_item_ids,
            remaining_item_ids=state.remaining_item_ids,
            evidence_chunk_ids=state.evidence_chunk_ids,
            citations=state.citations,
            created_at=effective_now,
            updated_at=effective_now,
            expires_at=expires_at,
        )
        await self._add_record(db=db, record=record)
        await self._commit(db)
        await self._refresh(db=db, record=record)
        return state

    async def load_active(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
        now: datetime | None = None,
    ) -> ContinuationState | None:
        """Load the active continuation state for a workspace/session."""

        if not self._enabled:
            return None

        effective_now = now or datetime.utcnow()
        record = await self._select_active_record(
            db=db,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        if record is None:
            return None

        if record.expires_at <= effective_now:
            record.status = AgenticSessionStatus.EXPIRED
            record.updated_at = effective_now
            await self._commit(db)
            return None

        try:
            return ContinuationState.model_validate(record.state)
        except Exception:
            record.status = AgenticSessionStatus.EXPIRED
            record.updated_at = effective_now
            await self._commit(db)
            return None

    async def mark_completed(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
        now: datetime | None = None,
    ) -> bool:
        """Mark the active continuation state as completed."""

        if not self._enabled:
            return False

        effective_now = now or datetime.utcnow()
        record = await self._select_active_record(
            db=db,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        if record is None:
            return False

        record.status = AgenticSessionStatus.COMPLETED
        record.remaining_item_ids = []
        record.state = {**record.state, "remaining_item_ids": []}
        record.updated_at = effective_now
        await self._commit(db)
        return True

    async def expire_old_states(
        self,
        *,
        db: AsyncSession,
        now: datetime | None = None,
    ) -> int:
        """Expire active records older than their TTL."""

        if not self._enabled:
            return 0

        effective_now = now or datetime.utcnow()
        records = await self._select_expired_records(db=db, now=effective_now)
        for record in records:
            record.status = AgenticSessionStatus.EXPIRED
            record.updated_at = effective_now
        if records:
            await self._commit(db)
        return len(records)

    @classmethod
    def is_continuation_intent(cls, message: str) -> bool:
        """Detect whether a user asks to continue unfinished work."""

        normalized = message.strip().casefold()
        if not normalized:
            return False
        return any(re.search(pattern, normalized) for pattern in cls._CONTINUATION_PATTERNS)

    async def _select_active_record(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
    ) -> AgenticSession | None:
        result = await db.execute(
            select(AgenticSession)
            .where(
                AgenticSession.workspace_id == workspace_id,
                AgenticSession.session_id == session_id,
                AgenticSession.status == AgenticSessionStatus.ACTIVE,
            )
            .order_by(AgenticSession.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _select_expired_records(
        self,
        *,
        db: AsyncSession,
        now: datetime,
    ) -> list[AgenticSession]:
        result = await db.execute(
            select(AgenticSession).where(
                AgenticSession.status == AgenticSessionStatus.ACTIVE,
                AgenticSession.expires_at <= now,
            )
        )
        return list(result.scalars().all())

    async def _expire_active_records(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
        now: datetime,
    ) -> None:
        records = await self._select_active_records(
            db=db,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        for record in records:
            record.status = AgenticSessionStatus.EXPIRED
            record.updated_at = now

    async def _select_active_records(
        self,
        *,
        db: AsyncSession,
        workspace_id: int,
        session_id: str,
    ) -> list[AgenticSession]:
        result = await db.execute(
            select(AgenticSession).where(
                AgenticSession.workspace_id == workspace_id,
                AgenticSession.session_id == session_id,
                AgenticSession.status == AgenticSessionStatus.ACTIVE,
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def _add_record(*, db: AsyncSession, record: AgenticSession) -> None:
        db.add(record)

    @staticmethod
    async def _commit(db: AsyncSession) -> None:
        await db.commit()

    @staticmethod
    async def _refresh(*, db: AsyncSession, record: AgenticSession) -> None:
        await db.refresh(record)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped


__all__ = ["ContinuationManager"]
