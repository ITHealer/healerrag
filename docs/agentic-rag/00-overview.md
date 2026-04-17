# Agentic RAG Upgrade Overview

Version: 1.0
Date: 2026-04-15
Status: Draft for implementation planning

## Purpose

This document set describes how to upgrade the current HealerRAG backend from a stable Hybrid RAG system into a safer Agentic RAG system.

The current system already works well:

- FastAPI backend with `/api/v1/rag` and `/api/v1/documents`.
- Document parsing through Docling or Marker.
- Vector retrieval through ChromaDB.
- Knowledge Graph storage through LightRAG.
- Reranking through `BAAI/bge-reranker-v2-m3`.
- SSE streaming chat with source and image events.

The new work must preserve that behavior while adding an optional agentic orchestration layer.

## Design Rule

All new behavior must be behind this feature flag:

```env
AGENTIC_RAG_ENABLED=false
```

When `AGENTIC_RAG_ENABLED=false`, the system must behave like the current implementation.

Web search must also be separately gated:

```env
AGENTIC_WEB_SEARCH_ENABLED=false
```

When web search is disabled, no provider-backed web call may be made even if API keys exist.

## Document Map

- `01-prd.md`: Product requirements and expected behavior.
- `02-sdd.md`: Software design details, components, interfaces, and integration points.
- `03-implementation-checklist.md`: Phase-by-phase implementation checklist for junior developers.

## Current Codebase Anchors

Use these existing files as anchors:

- `backend/app/api/rag.py`: REST endpoints for query, process, chat, debug, KG stats.
- `backend/app/api/chat_agent.py`: SSE streaming chat and semi-agentic tool calling.
- `backend/app/services/rag_service.py`: service factory and legacy RAG service.
- `backend/app/services/agentic_rag_service.py`: current HealerRAG processing and hybrid retrieval service.
- `backend/app/services/deep_retriever.py`: hybrid vector + KG retrieval with reranking.
- `backend/app/services/vector_store.py`: ChromaDB collection wrapper.
- `backend/app/services/knowledge_graph_service.py`: LightRAG wrapper.
- `backend/app/services/llm/`: provider abstraction for Gemini and Ollama.
- `backend/app/core/config.py`: Pydantic settings.

## Naming Decision

The file `backend/app/services/agentic_rag_service.py` already exists and handles document processing plus hybrid retrieval. Do not move or overload it with the new orchestrator.

Create the new orchestration package here:

```text
backend/app/services/agentic/
```

The new orchestrator class should be named:

```text
AgenticRAGOrchestrator
```

This avoids confusion between:

- `AgenticRAGService`: current processing/retrieval service.
- `AgenticRAGOrchestrator`: new multi-step reasoning, judging, budgeting, and continuation coordinator.

## Rollout Strategy

The upgrade must be implemented in phases:

1. Add models, config, and prompts.
2. Add deterministic components first.
3. Add LLM-judge components.
4. Add web search tool behind its own flag.
5. Add orchestrator.
6. Integrate with API only after all component tests pass.
7. Enable in local/dev only.
8. Keep production default disabled until manually approved.

## Key Safety Rules

- Never change document parser behavior as part of this feature.
- Never change ChromaDB metadata schema for existing chunks.
- Never change LightRAG storage format.
- Never break the current frontend event contract.
- Never cite web search results as if they were document chunks.
- Never run unbounded retrieval/rewrite/judge loops.
- Always degrade to current Hybrid RAG when agentic orchestration fails.
