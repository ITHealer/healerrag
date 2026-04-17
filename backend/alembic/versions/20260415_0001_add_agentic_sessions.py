"""add_agentic_sessions

Revision ID: 20260415_0001
Revises: 1d7b6141c8fd
Create Date: 2026-04-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260415_0001"
down_revision: Union[str, Sequence[str], None] = "1d7b6141c8fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "agentic_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("completed_item_ids", sa.JSON(), nullable=False),
        sa.Column("remaining_item_ids", sa.JSON(), nullable=False),
        sa.Column("evidence_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agentic_sessions_id"), "agentic_sessions", ["id"], unique=False)
    op.create_index(
        op.f("ix_agentic_sessions_workspace_id"),
        "agentic_sessions",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agentic_sessions_session_id"),
        "agentic_sessions",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agentic_sessions_status"),
        "agentic_sessions",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agentic_sessions_expires_at"),
        "agentic_sessions",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_agentic_sessions_workspace_session_status",
        "agentic_sessions",
        ["workspace_id", "session_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index("ix_agentic_sessions_workspace_session_status", table_name="agentic_sessions")
    op.drop_index(op.f("ix_agentic_sessions_expires_at"), table_name="agentic_sessions")
    op.drop_index(op.f("ix_agentic_sessions_status"), table_name="agentic_sessions")
    op.drop_index(op.f("ix_agentic_sessions_session_id"), table_name="agentic_sessions")
    op.drop_index(op.f("ix_agentic_sessions_workspace_id"), table_name="agentic_sessions")
    op.drop_index(op.f("ix_agentic_sessions_id"), table_name="agentic_sessions")
    op.drop_table("agentic_sessions")
