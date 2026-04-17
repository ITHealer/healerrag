from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — inject async DB session vào route handlers.
    
    Session tự động close khi request kết thúc (kể cả khi có exception).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()  # ← Rollback nếu có lỗi
            raise
        # Session tự đóng khi exit context manager