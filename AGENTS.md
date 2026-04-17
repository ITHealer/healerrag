# Repository Guidelines

## Project Shape

- Root workspace: `D:\ai-leanrning\production-rag`
- This is a FastAPI-based RAG backend with a currently empty `frontend/` directory.
- Backend source lives in `backend/app/`.
- API entrypoint: `backend/app/main.py`.
- Aggregated API router: `backend/app/api/router.py`, mounted under `settings.API_V1_PREFIX` which defaults to `/api/v1`.
- Database models are in `backend/app/models/`; Pydantic schemas are in `backend/app/schemas/`.
- Business logic and integrations are in `backend/app/services/`.
- Alembic migrations live in `backend/alembic/`.
- Uploaded local documents live under `backend/uploads/`; treat uploaded PDFs and generated artifacts as user data.

## Runtime And Dependencies

- Python requirement is `>=3.11`.
- Dependency metadata exists in both `backend/pyproject.toml` and `backend/requirements.txt`; prefer `uv` commands when working from `backend/` because `backend/uv.lock` is present.
- Root `.env` is loaded by `backend/app/core/config.py`; do not print or expose its contents.
- PostgreSQL and ChromaDB sidecars are defined in `docker-compose.services.yml`.

Useful commands:

```powershell
docker compose -f docker-compose.services.yml up -d
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

Health endpoints:

- `GET /health`
- `GET /ready`

Main API groups under `/api/v1`:

- `/workspaces`
- `/documents`
- `/rag`
- `/config`

## Validation

- No dedicated test suite or pytest configuration was present at initialization time.
- Before finishing backend changes, run the narrowest meaningful check available. At minimum, prefer:

```powershell
cd backend
uv run python -m compileall app
```

- If a test suite is added later, run the relevant tests for the touched area before finalizing.

## Implementation Notes

- Preserve the existing async SQLAlchemy pattern from `backend/app/core/database.py`.
- Prefer existing service factories, especially `get_rag_service()` and LLM provider helpers, over creating parallel integration paths.
- Do not normalize or rewrite Vietnamese comments or README text unless the task is specifically about encoding or documentation cleanup.
- Keep API changes aligned with the schemas in `backend/app/schemas/`.
- Be careful with background tasks that use their own DB session; verify the session factory name before editing those flows.
- Image and document paths can point at extracted/generated user content. Avoid deleting local uploads unless the user explicitly asks.

## Known Gotchas From Initialization Scan

- The workspace root was not a Git repository during initialization, so use extra care when reporting changed files.
- `backend/app/api/documents.py` references `async_session_maker`, while `backend/app/core/database.py` defines `AsyncSessionLocal` and `async_session_marker`. Verify and fix deliberately before relying on document background processing.
- `backend/app/api/rag.py` references `settings.OLLAMA_ENABLE_THINKING`, but that setting was not defined in `backend/app/core/config.py` during initialization. Check this before working on Ollama capability reporting.
- `frontend/` existed but contained no visible files during initialization.
