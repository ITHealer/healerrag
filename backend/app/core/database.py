from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# Không dùng create_engine thay create_async_engine vì Sync engine → block async event loop → hiệu suất kém, lỗi "RuntimeError: This event loop is already running"

# Database setup with SQLAlchemy AsyncIO 
# → Base là "bản mẫu gốc" mà tất cả các Model phải kế thừa
# Base.metadata biết tất cả bảng: users, posts,...
# Alembic dùng Base.metadata để so sánh và tạo migration
class Base(DeclarativeBase):
    pass

# Engine - connection pool đến PostgreSQL, sử dụng asyncpg driver
# →  Hệ thống đường truyền đến kho hàng (PostgreSQL)
engine = create_async_engine( 
    settings.DATABASE_URL,
    echo=settings.DEBUG,    # Log SQL queries nếu DEBUG=True
    pool_size=10,           # Số lượng kết nối tối đa trong pool
    max_overflow=20,        # Số lượng kết nối tối đa có thể tạo thêm khi pool đầy
    pool_pre_ping=True,     # Kiểm tra kết nối trước khi sử dụng (giúp tránh lỗi "connection closed")
)
"""
Tình huống không có pre_ping:
  - Kết nối nằm trong pool 10 phút không dùng
  - PostgreSQL tự đóng kết nối đó
  - Request mới lấy kết nối "chết" từ pool → LỖI 💥

Có pre_ping:
  - Trước khi dùng → gửi "SELECT 1" để test
  - Nếu chết → tự tạo kết nối mới → an toàn ✅
"""

# Session factory - tạo AsyncSession objects để tương tác với DB
# →  Máy tạo phiếu mua hàng (Session)
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession,
    expire_on_commit=False,  # ← QUAN TRỌNG: object không bị expire sau commit 
                             # Nếu True → access attribute sau commit → lazy load → lỗi async
)

# Alias for background tasks
# Trong HTTP request → dùng get_db()
# Trong background task → dùng async_session_marker
async_session_marker = AsyncSessionLocal


# →  Nhân viên thu ngân cho từng khách (mỗi request)
async def get_db() -> AsyncSession:
    """Dependency - tạo AsyncSession mới cho mỗi request, tự động đóng sau khi xong"""
    async with AsyncSessionLocal() as session:
        try: 
            yield session
        finally:
            await session.close()

"""
Flow của một HTTP request:
```
Client gửi request
       ↓
FastAPI gọi get_db()
       ↓
Tạo Session mới từ pool
       ↓
yield session ──────────────→ Route handler dùng session
                                  ↓
                              await session.execute(...)
                              await session.commit()
                                  ↓
◄──────────────────────────── Trả session về
       ↓
finally: session.close()
(trả connection về pool, không phải đóng hẳn)
       ↓
Trả response về client
```
"""