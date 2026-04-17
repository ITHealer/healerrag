
```
healer-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          ← FastAPI app + lifespan
│   │   ├── core/
│   │   │   ├── config.py    ← Pydantic Settings
│   │   │   ├── database.py  ← SQLAlchemy engine
│   │   │   ├── deps.py      ← DI dependencies
│   │   │   └── exceptions.py
│   │   ├── models/          ← SQLAlchemy ORM models
│   │   ├── schemas/         ← Pydantic request/response
│   │   ├── api/             ← Route handlers
│   │   └── services/        ← Business logic
│   ├── alembic/             ← DB migrations
│   └── requirements.txt
├── frontend/
├── docker-compose.yml
└── .env.example
```