# Backend Working Guide

## Scope

- File nay ap dung cho toan bo thu muc `backend/`.
- Day la living document cho backend. Moi thay doi backend trong tuong lai phai cap nhat lai file nay trong cung task neu thay doi do anh huong den:
  - cau truc folder hoac file
  - route API, schema request/response, model DB
  - service flow, parser, retrieval, chat, agentic flow
  - config, env var, dependency, command run/test
  - convention dat ten, pattern to chuc code
- Khong xem viec cap nhat `backend/AGENTS.md` la tuy chon. Day la mot phan cua definition of done cho backend.

## Muc tieu code

- Uu tien code de doc, de debug, de maintain hon la "clever".
- Viet nhu mot senior Python engineer, nhung uu tien cu phap pho bien, thang va de hieu khi co the.
- To chuc code sao cho junior dev co the lan theo tu route -> schema -> service -> model ma khong bi lac.
- Ten file, ten bien, ten ham, ten class phai mo ta dung y nghia nghiep vu.

## Tong quan cau truc

```text
backend/
|-- AGENTS.md
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- uv.lock
|-- alembic.ini
|-- app/
|   |-- main.py
|   |-- schema.sql
|   |-- api/
|   |-- core/
|   |-- models/
|   |-- schemas/
|   |-- services/
|-- alembic/
|   |-- env.py
|   `-- versions/
`-- tests/
    `-- agentic/
```

## Runtime va generated folders

- `.venv/`, `.uv-cache/`, `__pycache__/` la local/runtime artifact, khong xem la app architecture.
- `uploads/` la user data upload, khong xoa neu chua co yeu cau ro rang.
- `data/docling/` la shared static root cho anh trich xuat tu tai lieu, duoc phuc vu qua `/static/doc-images`.
- Cac parser co the co working dir rieng trong `data/<parser_name>/`, nhung image file phuc vu API/frontend phai duoc ghi vao shared static root de URL giu on dinh.
- `data/lightrag/` la du lieu KG theo workspace.

## Vai tro tung khu vuc

### Root files

- `pyproject.toml`: metadata va dependency chinh cho Python backend.
- `requirements.txt`: dependency list phu tro hoac compatibility.
- `uv.lock`: lockfile cho `uv`.
- `README.md`: huong dan khoi dong va tong quan backend.
- `alembic.ini`: config Alembic.
- `AGENTS.md`: ban do backend + coding rules + quy tac cap nhat bat buoc.

### `app/`

- `main.py`: tao FastAPI app, lifespan startup/shutdown, health/ready, mount API router, mount static image, auto recovery document processing stale.
- `schema.sql`: schema fallback de bootstrap DB khi can.

### `app/api/`

- `router.py`: aggregated router. Moi module API moi phai duoc mount tai day neu muon public.
- `workspaces.py`: CRUD workspace/knowledge base, thong ke tong hop, cleanup vector/KG/image khi xoa workspace.
- `documents.py`: upload, list, get, delete document; doc markdown/images; trigger background processing; validate file type va file size.
- Upload validation phai follow parser dang active thong qua `supported_extensions()`, khong hard-code mot danh sach extension co dinh de tranh lech voi `.env`.
- `rag.py`: query, process, reindex, stats, analytics, chat, capabilities; la HTTP surface chinh cho retrieval va chat.
- `config.py`: tra thong tin provider/model dang active cho frontend.
- `chat_prompt.py`: chua prompt constants va prompt text dung chung cho luong chat.
- `chat_agent.py`: module SSE/semi-agentic chat va tool-calling helper. Hien tai khong duoc mount trong `api/router.py`, nen can xac nhan truoc khi xem la endpoint production.

### `app/core/`

- `config.py`: Pydantic settings, load `.env`, feature flags, provider config, retrieval tuning, agentic toggles.
- `database.py`: async SQLAlchemy engine, session factory, `get_db`, alias session cho background task.
- `deps.py`: dependency helpers dung chung cho FastAPI routes.
- `exceptions.py`: custom exception va helper cho API layer.

### `app/models/`

- `knowledge_base.py`: model workspace/knowledge base.
- `document.py`: model document, image, table va enum status xu ly.
- `chat_message.py`: luu lich su chat va metadata chat theo workspace.
- `agentic_session.py`: luu continuation state cho agentic RAG.
- `__init__.py`: import registry cho SQLAlchemy metadata.

### `app/schemas/`

- `workspace.py`: request/response schema cho workspace.
- `document.py`: request/response schema cho document upload va document metadata.
- `rag.py`: schema cho query, retrieval result, chat, analytics, capabilities va payload lien quan.
- Rule: khi doi API contract thi schema phai duoc cap nhat dong bo, khong de route tra payload "ngam hieu".

### `app/services/`

- `rag_service.py`: legacy/simple RAG service va factory `get_rag_service()`.
- `agentic_rag_service.py`: HEALERRAG pipeline day du cho parse -> dedup -> index -> hybrid retrieval.
- `chunking/`: nhom logic chunking.
- `chunking/chunker.py`: chunk text cho embedding/retrieval.
- `chunking/chunk_dedup.py`: loai bo chunk exact/near-duplicate truoc khi index.
- `embeddings/`: nhom logic embedding.
- `embeddings/embedder.py`: embedding service implementation.
- `embeddings/factory.py`: shared factory helpers cho embedding service.
- `loader/document_loader.py`: loader co ban cho txt/pdf/md.
- `retrieval/`: nhom retrieval stack.
- `retrieval/vector_store.py`: wrapper cho ChromaDB, namespace theo workspace, query/add/delete collection.
- `retrieval/reranker.py`: cross-encoder reranker service.
- `retrieval/deep_retriever.py`: hybrid retrieval, ket hop KG + vector + reranker + image/table refs.
- `knowledge_graph/knowledge_graph_service.py`: LightRAG KG service theo workspace.
- `shared/`: namespace cho utility/service dung chung (hien dang de san cho mo rong).

### `app/services/document_parser/`

- `base.py`: abstract parser contract.
- `docling_parser.py`: parser uu tien cho rich extraction.
- `marker_parser.py`: parser thay the nhe hon, tot hon cho mot so tai lieu co cong thuc.
- `mineru_parser.py`: parser dung official MinerU CLI + MinerU2.5-Pro-2604-1.2B, phu hop cho parse da dinh dang va giu model/cache/config nam trong `backend/models/mineru_models`.
- `marker_parser.py` co `__main__` de test parser local va export artifacts (markdown/chunks/images/tables/summary) ra folder chi dinh.
- `mineru_parser.py` co `__main__` de test parser local, download model ve project, va export artifacts (markdown/chunks/images/tables/summary) ra folder chi dinh.
- `__init__.py`: parser factory.
- Image extraction rule: file phuc vu qua API phai dung shared served path cho workspace, khong tro thang vao raw artifact path cua parser engine.

### `app/services/llm/`

- `base.py`: abstract interface cho LLM provider va embedding provider.
- `gemini.py`: Gemini provider implementation.
- `ollama.py`: Ollama provider implementation.
- `sentence_transformer.py`: local embedding provider implementation.
- `types.py`: typed payload/message/chunk objects cho llm layer.
- `__init__.py`: provider factory.

### `app/services/models/`

- `parsed_document.py`: internal typed models cho parsed document, citations, deep retrieval result.

### `app/services/agentic/`

- `models.py`: typed state/model cho agentic pipeline.
- `orchestrator.py`: coordinator chinh cho agentic flow.
- `query_analyzer.py`: phan tich do phuc tap query.
- `response_planner.py`: chia execution plan.
- `parallel_retrieval.py`: retrieval orchestration theo sub-query.
- `sufficiency_judge.py`: danh gia retrieval da du hay chua.
- `query_rewriter.py`: rewrite query sau khi judge fail.
- `hierarchical_synthesizer.py`: tong hop ket qua theo nhieu lop.
- `response_judge.py`: cham quality response.
- `continuation_manager.py`: luu/khai thac continuation state.
- `context_budget_manager.py`: cat/chon context theo budget.
- `web_search_tool.py`: web search adapter cho agentic flow.
- `observability.py`: log metadata va telemetry helper.

### `alembic/`

- `env.py`: wiring metadata/migration environment.
- `versions/`: migration files. Moi thay doi model DB can co migration tuong ung, tru khi user chi ro khong can.

### `tests/`

- `tests/agentic/`: test cho agentic modules.
- Khi mo rong test suite, uu tien dat test gan nghia vu module va ten file theo `test_<feature>.py`.

## Luong code nen theo

1. `api/` nhan request va validate input o muc HTTP.
2. `schemas/` dinh nghia hop dong request/response ro rang.
3. `services/` chua business logic va integration.
4. `models/` chua persisted state trong database.
5. `core/` chua config, database, dependency, exception infrastructure.

Rule:

- Route nen mong, service nen giai quyet logic.
- Neu route bat dau chua qua nhieu branching, query DB, transformation, hoac retry logic, hay day bot xuong service/helper.
- Khong de business rule quan trong bi lap lai o nhieu route.

## Quy tac dat ten

- File dung `snake_case.py`.
- Class dung `PascalCase`.
- Function/variable dung `snake_case`.
- Ten phai uu tien day du nghia:
  - dung `workspace_id`, `document_id`, `session_id`
  - dung `retrieved_chunks`, `document_status`, `metadata_filter`
  - tranh dat ten chung chung nhu `data`, `info`, `temp`, `obj`, `x`
- Collection dung so nhieu: `documents`, `chunks`, `image_refs`.
- Boolean dung tien to ro nghia: `is_`, `has_`, `should_`, `enable_`, `allow_`.
- Chi chap nhan viet tat pho bien va da ro nghia trong codebase: `db`, `llm`, `kg`, `api`.
- Neu mot bien la path, dat ten co hau to `_path`; neu la dir, dat ten `_dir`; neu la id, dat ten `_id`.

## Clean code rules

- Uu tien early return thay vi nest qua sau.
- Ham nen co mot muc dich ro rang. Neu mot ham vua validate, vua query, vua format, vua side-effect, can tach nho.
- Public function/class nen co type hint.
- Comment chi nen giai thich "vi sao", khong can mo ta lai dieu code da noi ro.
- Log phai co context de debug duoc, nhung khong log secret, token, hoac toan bo `.env`.
- Khong dua logic "magic" vao default value ma khong noi ro.
- Tranh abstraction qua muc, meta-programming, hoac one-liner kho doc neu loi ich khong ro rang.
- Uu tien control flow pho bien, de doc:
  - `if/else` ro rang hon expression qua phuc tap
  - loop ro rang hon nested comprehension kho doc
  - helper nho ro ten hon copy-paste logic
- Giu import nhat quan: stdlib -> third-party -> local.

## Async, DB, va background task

- Giu pattern async SQLAlchemy hien tai tu `app/core/database.py`.
- HTTP request dung `get_db()`.
- Background task phai dung session factory alias hien hanh trong `database.py`, khong tu tao pattern rieng.
- Neu can concurrent DB work, xac nhan session co cho phep; khong dung chung mot `AsyncSession` cho nhieu thao tac song song.

## API, schema, va persistence rules

- Moi thay doi response payload phai cap nhat schema lien quan.
- Moi thay doi model DB phai xem xet migration Alembic.
- Moi route moi phai co:
  - schema request/response neu la public contract
  - wiring trong `app/api/router.py` neu can expose
  - validation/check toi thieu phu hop
- Neu them env var moi:
  - khai bao trong `app/core/config.py`
  - dung ten ro rang, nhat quan voi settings hien co
  - cap nhat `backend/AGENTS.md` neu env do doi architecture hoac runbook
- MinerU runbook:
  - `HEALERRAG_DOCUMENT_PARSER=mineru` de bat parser MinerU.
  - `HEALERRAG_MINERU_BACKEND` default la `vlm-auto-engine`.
  - `HEALERRAG_MINERU_SOURCE` default la `local` de uu tien model duoc quan ly ben trong project.
  - `HEALERRAG_MINERU_MODELS_DIR` default tro toi `backend/models/mineru_models`.
  - `HEALERRAG_MINERU_MODEL_ID` default la `opendatalab/MinerU2.5-Pro-2604-1.2B`.
  - `HEALERRAG_MINERU_PIPELINE_MODEL_DIR` la optional, chi can khi muon chay `pipeline`/`hybrid-*` bang source local.
  - MinerU config runtime duoc ghi ra `backend/models/mineru_models/mineru.json` va env `MINERU_TOOLS_CONFIG_JSON` phai tro toi file nay.
  - Hugging Face cache cho MinerU phai nam trong `backend/models/mineru_models/.hf-cache`, khong de model/cache roi ra ngoai project neu chua co ly do van hanh ro rang.

## HealerRAG va retrieval rules

- Uu tien tai su dung `get_rag_service()` thay vi mo them mot path retrieval song song neu chua that su can.
- Neu thay doi parser, chunking, dedup, rerank, KG, hoac vector flow, phai cap nhat lai phan mo ta service trong file nay.
- Can than voi uploads, extracted images, va generated artifacts vi day la user data.
- Khong xoa local uploads, doc images, hoac KG data neu user chua yeu cau hoac business flow khong doi hoi ro rang.

## Validation toi thieu

- Truoc khi chot thay doi backend, chay check hep nhat co y nghia cho phan vua sua.
- Toi thieu nen co compile check:

```powershell
cd backend
uv run python -m compileall app
```

- Neu `uv` gap van de local packaging/environment, duoc phep dung direct Python compile check de xac nhan syntax, nhung khong bo qua validation ma khong noi ro ly do.
- Neu da sua agentic modules, uu tien chay test lien quan trong `tests/agentic/`.
- CLI test parser local:

```powershell
cd backend
python -m app.services.document_parser.mineru_parser "D:\path\to\document.pdf" --output-dir "D:\path\to\mineru_outputs"
```

- Download truoc model MinerU vao trong project:

```powershell
cd backend
python -m app.services.document_parser.mineru_parser --output-dir "D:\tmp\mineru" --download-model-only
```

## Khi nao bat buoc cap nhat file nay

Bat buoc sua `backend/AGENTS.md` trong cung task neu:

- them/xoa/doi ten file hoac folder trong `backend/`
- them route moi, doi endpoint behavior, doi response contract
- doi model, schema, migration, session pattern, service factory
- doi provider, parser, reranker, KG, vector, chat, agentic flow
- doi command setup/run/test quan trong
- doi coding convention ma team can tuan theo

Neu thay doi backend ma khong can sua file nay, phai co ly do ro rang va thuc su chi la thay doi cuc bo khong anh huong den architecture, conventions, hoac runbook.
