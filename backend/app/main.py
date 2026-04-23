from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text, update

import logging
import os

from app.core.config import settings
from app.core.database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Chạy khi server START (trước yield) và STOP (sau yield).
    Thay thế @on_event("startup") đã deprecated.
    """
    logger.info("🚀 Starting HealerRAG API...")


    # --- Bước 1: Auto-create database tables (nếu chưa có) ---
    auto_create = os.getenv("AUTO_CREATE_TABLES", "true").lower() == "true"
    if auto_create:
        async with engine.begin() as conn:
            # Check if tables already exist (e.g., alembic_version) tránh tạo lại nếu đã có
            result = await conn.execute(text(
                "SELECT EXISTS ("
                "  SELECT FROM information_schema.tables "
                "  WHERE table_schema = 'public' "
                "  AND table_name = 'alembic_version'"
                ");"
            ))
            is_tables_exist = result.scalar()

            if not is_tables_exist:
                logger.info("No tables found. Creating database tables...")
                
                schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
                if os.path.exists(schema_path):
                    # Dùng schema.sql để tạo bảng thay vì Base.metadata.create_all() để tránh lỗi async
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema_sql = f.read()

                    # asyncpg không support multi-statement → split và execute từng cái
                    for statement in schema_sql.split(';'):
                        stmt = statement.strip()
                        if stmt:
                            await conn.execute(text(stmt))

                    # Stamp alembic version để skip migration lần sau
                    await conn.execute(text(
                        "INSERT INTO public.alembic_version (version_num) "
                        "VALUES ('initial') ON CONFLICT DO NOTHING;"
                    ))
                    logger.info("✅ Database tables created from schema.sql")
                else:
                    # Fallback: SQLAlchemy auto-create (dev only)
                    await conn.run_sync(Base.metadata.create_all)
                    logger.info("✅ Database tables created via SQLAlchemy")
            else:
                logger.info("✅ Database already initialized — skipping")


        # --- Bước 2: Recover stale processing documents ---
        # Documents bị stuck ở PARSING/INDEXING từ lần run trước (crash/restart)
        from app.models.document import Document, DocumentStatus
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy import select as sa_select
        async with AsyncSession(engine) as session:
            timeout = settings.HEALERRAG_PROCESSING_TIMEOUT_MINUTES
            cutoff = datetime.utcnow() - timedelta(minutes=timeout)
            stale_statuses = [
                DocumentStatus.PROCESSING,
                DocumentStatus.PARSING,
                DocumentStatus.INDEXING,
            ]
            result = await session.execute(
                update(Document)
                .where(
                    Document.status.in_(stale_statuses),
                    Document.updated_at < cutoff,
                )
                .values(
                    status=DocumentStatus.FAILED,
                    error_message=f"Processing timeout ({timeout}min). Click Analyze to retry.",
                )
                .returning(Document.id)
            )
            stale_ids = [row[0] for row in result.fetchall()]
            if stale_ids:
                await session.commit()
                logger.warning(f"Recovered {len(stale_ids)} stale documents: {stale_ids}")
    else:
        logger.info("AUTO_CREATE_TABLES=false — skipping auto-migration")


    logger.info("✅ HealerRAG API ready!")
    yield   # ← Server chạy từ đây

    # ── Cleanup khi shutdown ──────────────────────────────────────────────
    logger.info("🛑 Shutting down HealerRAG...")
    await engine.dispose()   # Đóng tất cả DB connections


# --- FastAPI app setup ---
app = FastAPI(
    title=settings.APP_NAME,
    description="API for HealerRAG - Knowledge Base with semantic search, knowledge graph, and LLM chatbot",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,   # Dùng lifespan thay vì on_event
    docs_url="/docs",    # Swagger UI
    redoc_url="/redoc",  # ReDoc UI
    redirect_slashes=False, # /workspaces và /workspaces/ = khác nhau, không redirect
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/ready", tags=["System"])
async def ready():
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}



# --- API routes ---
from app.api.router import api_router  # noqa: E402
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Static files — document images extracted by parsers
_docling_data = Path(__file__).resolve().parent.parent / "data" / "docling"
_docling_data.mkdir(parents=True, exist_ok=True)
app.mount("/static/doc-images", StaticFiles(directory=str(_docling_data)), name="static_doc_images")

# Import models so SQLAlchemy registers them
from app.models import knowledge_base, document, chat_message
