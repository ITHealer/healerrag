from datetime import datetime, timedelta
import unittest

from app.models.agentic_session import AgenticSession, AgenticSessionStatus
from app.services.agentic.continuation_manager import ContinuationManager
from app.services.agentic.models import ExecutionItem, ExecutionPlan


class InMemoryDB:
    def __init__(self) -> None:
        self.records: list[AgenticSession] = []
        self.next_id = 1
        self.commits = 0


class InMemoryContinuationManager(ContinuationManager):
    async def _select_active_record(
        self,
        *,
        db: InMemoryDB,
        workspace_id: int,
        session_id: str,
    ) -> AgenticSession | None:
        records = await self._select_active_records(
            db=db,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        if not records:
            return None
        return sorted(records, key=lambda record: record.updated_at, reverse=True)[0]

    async def _select_active_records(
        self,
        *,
        db: InMemoryDB,
        workspace_id: int,
        session_id: str,
    ) -> list[AgenticSession]:
        return [
            record
            for record in db.records
            if record.workspace_id == workspace_id
            and record.session_id == session_id
            and record.status == AgenticSessionStatus.ACTIVE
        ]

    async def _select_expired_records(
        self,
        *,
        db: InMemoryDB,
        now: datetime,
    ) -> list[AgenticSession]:
        return [
            record
            for record in db.records
            if record.status == AgenticSessionStatus.ACTIVE and record.expires_at <= now
        ]

    @staticmethod
    async def _add_record(*, db: InMemoryDB, record: AgenticSession) -> None:
        record.id = db.next_id
        db.next_id += 1
        db.records.append(record)

    @staticmethod
    async def _commit(db: InMemoryDB) -> None:
        db.commits += 1

    @staticmethod
    async def _refresh(*, db: InMemoryDB, record: AgenticSession) -> None:
        return None


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        can_fully_answer_now=False,
        total_items=3,
        items=[
            ExecutionItem(
                item_id="item_1",
                description="Revenue",
                priority=1,
                estimated_output_tokens=300,
                related_sub_queries=["revenue"],
            ),
            ExecutionItem(
                item_id="item_2",
                description="Margin",
                priority=2,
                estimated_output_tokens=300,
                related_sub_queries=["margin"],
            ),
            ExecutionItem(
                item_id="item_3",
                description="Free cash flow",
                priority=3,
                estimated_output_tokens=300,
                related_sub_queries=["fcf"],
            ),
        ],
        batch_now=["item_1"],
        batch_later=["item_2", "item_3"],
        continuation_message="Continue later.",
    )


class ContinuationManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_creates_active_state_with_schema_version(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager(ttl_hours=2)
        now = datetime(2026, 4, 15, 10, 0, 0)

        state = await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1", "item_1"],
            remaining_item_ids=["item_2", "item_3"],
            evidence_chunk_ids=["c1", "c1", "c2"],
            citations=[{"chunk_id": "c1"}],
            now=now,
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.schema_version, 1)
        self.assertEqual(state.completed_item_ids, ["item_1"])
        self.assertEqual(state.evidence_chunk_ids, ["c1", "c2"])
        self.assertEqual(len(db.records), 1)
        self.assertEqual(db.records[0].status, AgenticSessionStatus.ACTIVE)
        self.assertEqual(db.records[0].state["schema_version"], 1)
        self.assertEqual(db.records[0].expires_at, now + timedelta(hours=2))

    async def test_load_active_is_workspace_scoped(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager()
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now,
        )

        self.assertIsNone(await manager.load_active(db=db, workspace_id=2, session_id="chat-1", now=now))
        loaded = await manager.load_active(db=db, workspace_id=1, session_id="chat-1", now=now)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, "chat-1")
        self.assertEqual(loaded.remaining_item_ids, ["item_2"])

    async def test_save_expires_previous_active_state_for_same_workspace_session(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager()
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="First query",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now,
        )
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Second query",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_3"],
            evidence_chunk_ids=["c2"],
            citations=[],
            now=now + timedelta(minutes=1),
        )

        statuses = [record.status for record in db.records]
        self.assertEqual(statuses, [AgenticSessionStatus.EXPIRED, AgenticSessionStatus.ACTIVE])
        loaded = await manager.load_active(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            now=now + timedelta(minutes=1),
        )
        self.assertEqual(loaded.original_query, "Second query")

    async def test_expired_active_state_is_ignored_and_marked_expired(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager(ttl_hours=1)
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now,
        )

        loaded = await manager.load_active(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            now=now + timedelta(hours=2),
        )

        self.assertIsNone(loaded)
        self.assertEqual(db.records[0].status, AgenticSessionStatus.EXPIRED)

    async def test_mark_completed_hides_active_state(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager()
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now,
        )

        completed = await manager.mark_completed(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            now=now,
        )

        self.assertTrue(completed)
        self.assertEqual(db.records[0].status, AgenticSessionStatus.COMPLETED)
        self.assertEqual(db.records[0].remaining_item_ids, [])
        self.assertEqual(db.records[0].state["remaining_item_ids"], [])
        self.assertIsNone(await manager.load_active(db=db, workspace_id=1, session_id="chat-1", now=now))

    async def test_save_skips_when_no_remaining_items(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager()

        state = await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1", "item_2", "item_3"],
            remaining_item_ids=[],
            evidence_chunk_ids=["c1"],
            citations=[],
        )

        self.assertIsNone(state)
        self.assertEqual(db.records, [])

    async def test_corrupt_state_is_expired_on_load(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager()
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now,
        )
        db.records[0].state = {"bad": "state"}

        loaded = await manager.load_active(db=db, workspace_id=1, session_id="chat-1", now=now)

        self.assertIsNone(loaded)
        self.assertEqual(db.records[0].status, AgenticSessionStatus.EXPIRED)

    async def test_expire_old_states_expires_only_old_active_records(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager(ttl_hours=1)
        now = datetime(2026, 4, 15, 10, 0, 0)
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="old",
            original_query="Old",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
            now=now - timedelta(hours=2),
        )
        await manager.save(
            db=db,
            workspace_id=1,
            session_id="fresh",
            original_query="Fresh",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c2"],
            citations=[],
            now=now,
        )

        count = await manager.expire_old_states(db=db, now=now)

        self.assertEqual(count, 1)
        self.assertEqual(db.records[0].status, AgenticSessionStatus.EXPIRED)
        self.assertEqual(db.records[1].status, AgenticSessionStatus.ACTIVE)

    async def test_disabled_manager_does_not_persist_or_load(self) -> None:
        db = InMemoryDB()
        manager = InMemoryContinuationManager(enabled=False)
        state = await manager.save(
            db=db,
            workspace_id=1,
            session_id="chat-1",
            original_query="Analyze NVDA",
            execution_plan=_plan(),
            completed_item_ids=["item_1"],
            remaining_item_ids=["item_2"],
            evidence_chunk_ids=["c1"],
            citations=[],
        )

        self.assertIsNone(state)
        self.assertEqual(db.records, [])
        self.assertIsNone(await manager.load_active(db=db, workspace_id=1, session_id="chat-1"))
        self.assertFalse(await manager.mark_completed(db=db, workspace_id=1, session_id="chat-1"))
        self.assertEqual(await manager.expire_old_states(db=db), 0)

    def test_continuation_intent_detection(self) -> None:
        self.assertTrue(ContinuationManager.is_continuation_intent("continue"))
        self.assertTrue(ContinuationManager.is_continuation_intent("ti\u1ebfp t\u1ee5c ph\u1ea7n c\u00f2n l\u1ea1i"))
        self.assertTrue(ContinuationManager.is_continuation_intent("Lam tiep"))
        self.assertFalse(ContinuationManager.is_continuation_intent("what is NVDA revenue?"))


if __name__ == "__main__":
    unittest.main()
