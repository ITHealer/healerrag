from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Tìm .env file - check từ root project, fallback cho Docker context
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"

class Settings(BaseSettings):
    # App 
    APP_NAME: str = Field("My FastAPI App", env="APP_NAME")
    APP_VERSION: str = Field("0.1.0", env="APP_VERSION")
    DEBUG: bool = Field(False, env="DEBUG")
    API_V1_PREFIX: str = Field("/api/v1", env="API_V1_PREFIX")

    # Base directory (backend/ folder) - dùng để tính đường dẫn tuyệt đối
    BASE_DIR: Path = Field(Path(__file__).resolve().parent.parent.parent, env="BASE_DIR")

    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5002/healerrag", env="DATABASE_URL")

    # LLM Provider
    LLM_PROVIDER: str = Field("openai", env="LLM_PROVIDER") # "openai" | "gemini" | "ollama"

    # OpenAI
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")

    # Google Gemini
    GEMINI_API_KEY: str = Field("", env="GEMINI_API_KEY")   
    GOOGLE_AI_API_KEY: str = Field("", env="GOOGLE_AI_API_KEY")

    # Ollama
    OLLAMA_HOST: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    OLLAMA_MODEL: str = Field("gemma3:12b", env="OLLAMA_MODEL")
    OLLAMA_ENABLE_THINKING: bool = Field(False, env="OLLAMA_ENABLE_THINKING")

    # LLM (fast model for chat + KG extraction — used when provider=gemini)
    LLM_MODEL_FAST: str = Field("gemini-2.5-flash", env="LLM_MODEL_FAST")

    # Thinking level for Gemini 3.x+ models: "minimal" | "low" | "medium" | "high"
    # Gemini 2.5 uses thinking_budget_tokens instead (auto-detected)
    LLM_THINKING_LEVEL: str = Field("medium", env="LLM_THINKING_LEVEL") # "low" | "medium" | "high"

    # Max output tokens for LLM chat responses (includes thinking tokens)
    # Gemini 3.1 Flash-Lite supports up to 65536
    LLM_MAX_OUTPUT_TOKENS: int = Field(8192, env="LLM_MAX_OUTPUT_TOKENS")


    # KG Embedding provider (can differ from LLM provider)
    KG_EMBEDDING_PROVIDER: str = Field(default="gemini")
    KG_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")
    KG_EMBEDDING_DIMENSION: int = Field(default=3072)

    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)

    # HealerRAG Pipeline
    HEALERRAG_ENABLED: bool = True
    HEALERRAG_ENABLE_KG: bool = True
    HEALERRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    HEALERRAG_ENABLE_IMAGE_CAPTIONING: bool = True
    HEALERRAG_ENABLE_TABLE_CAPTIONING: bool = True
    HEALERRAG_MAX_TABLE_MARKDOWN_CHARS: int = 8000
    HEALERRAG_CHUNK_MAX_TOKENS: int = 512
    HEALERRAG_KG_QUERY_TIMEOUT: float = 30.0
    HEALERRAG_KG_CHUNK_TOKEN_SIZE: int = 1200
    HEALERRAG_KG_LANGUAGE: str = "English"
    HEALERRAG_KG_ENTITY_TYPES: list[str] = [
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ]
    HEALERRAG_DEFAULT_QUERY_MODE: str = "hybrid"
    HEALERRAG_DOCLING_IMAGES_SCALE: float = 2.0
    HEALERRAG_MAX_IMAGES_PER_DOC: int = 50 # Giới hạn ảnh tối đa mỗi file
    HEALERRAG_ENABLE_FORMULA_ENRICHMENT: bool = True

    # Document Parser provider: "docling" (default) or "marker" (lighter, better math)
    HEALERRAG_DOCUMENT_PARSER: str = "docling"
    HEALERRAG_MARKER_USE_LLM: bool = False

    # Processing timeout (minutes) — stale documents auto-recover to FAILED
    HEALERRAG_PROCESSING_TIMEOUT_MINUTES: int = 10

    # Pre-ingestion Deduplication
    HEALERRAG_DEDUP_ENABLED: bool = True
    HEALERRAG_DEDUP_MIN_CHUNK_LENGTH: int = 50       # min meaningful chars
    HEALERRAG_DEDUP_NEAR_THRESHOLD: float = 0.85     # Jaccard similarity cutoff

    # HealerRAG Retrieval Quality
    HEALERRAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    HEALERRAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    HEALERRAG_VECTOR_PREFETCH: int = 20
    HEALERRAG_RERANKER_TOP_K: int = 8
    HEALERRAG_MIN_RELEVANCE_SCORE: float = 0.15

    # Agentic RAG (opt-in; disabled by default for safe rollout)
    AGENTIC_RAG_ENABLED: bool = Field(False, env="AGENTIC_RAG_ENABLED")
    AGENTIC_RAG_MAX_RETRIEVAL_ATTEMPTS: int = Field(3, env="AGENTIC_RAG_MAX_RETRIEVAL_ATTEMPTS")
    AGENTIC_RAG_MAX_REPLAN_ATTEMPTS: int = Field(2, env="AGENTIC_RAG_MAX_REPLAN_ATTEMPTS")
    AGENTIC_RAG_REPLAN_ON_RESPONSE_FAIL: bool = Field(False, env="AGENTIC_RAG_REPLAN_ON_RESPONSE_FAIL")
    AGENTIC_RAG_SUFFICIENCY_THRESHOLD: float = Field(0.7, env="AGENTIC_RAG_SUFFICIENCY_THRESHOLD")
    AGENTIC_RAG_FAITHFULNESS_THRESHOLD: float = Field(0.7, env="AGENTIC_RAG_FAITHFULNESS_THRESHOLD")
    AGENTIC_RAG_COMPLETENESS_THRESHOLD: float = Field(0.7, env="AGENTIC_RAG_COMPLETENESS_THRESHOLD")
    AGENTIC_RAG_JUDGE_MODEL: str = Field("gemini-2.5-flash", env="AGENTIC_RAG_JUDGE_MODEL")
    AGENTIC_RAG_ANALYZER_MODEL: str = Field("gemini-2.5-flash", env="AGENTIC_RAG_ANALYZER_MODEL")
    AGENTIC_RAG_PLANNER_MODEL: str = Field("gemini-2.5-flash", env="AGENTIC_RAG_PLANNER_MODEL")

    # Agentic RAG performance and output budgets
    AGENTIC_PARALLEL_RETRIEVAL_TIMEOUT: float = Field(90.0, env="AGENTIC_PARALLEL_RETRIEVAL_TIMEOUT")
    AGENTIC_PARALLEL_RETRIEVAL_MAX_CONCURRENCY: int = Field(1, env="AGENTIC_PARALLEL_RETRIEVAL_MAX_CONCURRENCY")
    AGENTIC_JUDGE_TIMEOUT: float = Field(5.0, env="AGENTIC_JUDGE_TIMEOUT")
    AGENTIC_MAX_FINAL_CONTEXT_TOKENS: int = Field(5000, env="AGENTIC_MAX_FINAL_CONTEXT_TOKENS")
    AGENTIC_MAX_FINAL_CHUNKS: int = Field(8, env="AGENTIC_MAX_FINAL_CHUNKS")
    AGENTIC_MAX_CHUNKS_PER_SUBQUERY: int = Field(2, env="AGENTIC_MAX_CHUNKS_PER_SUBQUERY")
    AGENTIC_MAX_OUTPUT_TOKENS_PER_TURN: int = Field(1800, env="AGENTIC_MAX_OUTPUT_TOKENS_PER_TURN")
    AGENTIC_CONTINUATION_ENABLED: bool = Field(True, env="AGENTIC_CONTINUATION_ENABLED")
    AGENTIC_CONTINUATION_TTL_HOURS: int = Field(24, env="AGENTIC_CONTINUATION_TTL_HOURS")

    # Agentic RAG — Granular component toggles (bật/tắt từng thành phần để debug và A/B test)
    # ─────────────────────────────────────────────────────────────────────────────────────────
    # [1] Query Analysis: classify complexity (single_hop / multi_hop / no_retrieval)
    #     Nếu OFF → luôn coi là single_hop, dùng raw query trực tiếp
    AGENTIC_ENABLE_QUERY_ANALYSIS: bool = Field(True, env="AGENTIC_ENABLE_QUERY_ANALYSIS")

    # [2] Response Planning: chia sub-queries thành batch_now / batch_later
    #     Nếu OFF → 1 ExecutionPlan đơn giản với toàn bộ query trong batch_now
    AGENTIC_ENABLE_RESPONSE_PLANNER: bool = Field(True, env="AGENTIC_ENABLE_RESPONSE_PLANNER")

    # [3] Sufficiency Judgment: đánh giá kết quả retrieval đã đủ chưa (retry loop)
    #     Nếu OFF → chỉ retrieve 1 lần, không retry, không query rewrite
    AGENTIC_ENABLE_SUFFICIENCY_JUDGE: bool = Field(True, env="AGENTIC_ENABLE_SUFFICIENCY_JUDGE")

    # [4] Query Rewriting: rewrite query khi sufficiency judge fail
    #     Nếu OFF → sufficiency fail nhưng không rewrite, exit loop sớm
    AGENTIC_ENABLE_QUERY_REWRITER: bool = Field(True, env="AGENTIC_ENABLE_QUERY_REWRITER")

    # [5] Hierarchical Synthesis: LLM summarize từng sub-query trước khi generate
    #     Nếu OFF → bỏ qua LLM summarize, assemble trực tiếp từ raw chunks
    AGENTIC_ENABLE_HIERARCHICAL_SYNTHESIS: bool = Field(True, env="AGENTIC_ENABLE_HIERARCHICAL_SYNTHESIS")

    # [6] Response Judge: đánh giá chất lượng câu trả lời (faithfulness, completeness)
    #     Nếu OFF → bỏ qua judge, dùng generated_answer trực tiếp làm final_answer
    AGENTIC_ENABLE_RESPONSE_JUDGE: bool = Field(True, env="AGENTIC_ENABLE_RESPONSE_JUDGE")

    # Provider-backed web search (disabled by default; implemented in PR-02)
    AGENTIC_WEB_SEARCH_ENABLED: bool = Field(False, env="AGENTIC_WEB_SEARCH_ENABLED")
    AGENTIC_WEB_SEARCH_MAX_RESULTS: int = Field(5, env="AGENTIC_WEB_SEARCH_MAX_RESULTS")
    WEB_SEARCH_TOOL_BACKEND: str = Field("auto", env="WEB_SEARCH_TOOL_BACKEND")
    WEB_SEARCH_OPENAI_MODEL: str = Field("gpt-5.2", env="WEB_SEARCH_OPENAI_MODEL")
    WEB_SEARCH_GOOGLE_MODEL: str = Field("gemini-3-flash-preview", env="WEB_SEARCH_GOOGLE_MODEL")
    WEB_SEARCH_TIMEOUT_SECONDS: float = Field(30.0, env="WEB_SEARCH_TIMEOUT_SECONDS")
    WEB_SEARCH_MAX_QUERIES: int = Field(2, env="WEB_SEARCH_MAX_QUERIES")
    OPENAI_BASE_URL: str = Field("", env="OPENAI_BASE_URL")
    GOOGLE_BASE_URL: str = Field("", env="GOOGLE_BASE_URL")


    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:8008",
        "http://localhost:3000",
    ]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Sử dụng lru_cache để cache settings, tránh phải load nhiều lần
@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Tạo instance settings toàn cục, có thể import ở bất kỳ đâu trong app
settings = get_settings()
