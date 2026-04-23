from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

import numpy as np
from openai import OpenAI

from app.services.llm.base import LLMProvider, EmbeddingProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


# ------------------------------
# LLM Provider
# ------------------------------
class VLLMProvider(LLMProvider):
    """OpenAI-compatible vLLM provider."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.VLLM_DEFAULT_MODEL
        self.client = OpenAI(
            api_key=settings.VLLM_API_KEY,
            base_url=settings.VLLM_API_BASE_URL,
        )

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = settings.VLLM_TEMPERATURE,
        max_tokens: int = settings.VLLM_MAX_TOKENS,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            chat_messages.append({"role": msg.role, "content": msg.content})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            if think:
                return LLMResult(content=content, thinking="")
            return content
        except Exception as e:
            logger.error(f"vLLM complete() failed: {e}")
            return LLMResult(content="") if think else ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = settings.VLLM_TEMPERATURE,
        max_tokens: int = settings.VLLM_MAX_TOKENS,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        return await asyncio.to_thread(
            self.complete,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = settings.VLLM_TEMPERATURE,
        max_tokens: int = settings.VLLM_MAX_TOKENS,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        # vLLM chưa có streaming native → fallback: yield full response once
        result = await self.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )
        if isinstance(result, LLMResult):
            if result.thinking:
                yield StreamChunk(type="thinking", text=result.thinking)
            yield StreamChunk(type="text", text=result.content)
        else:
            yield StreamChunk(type="text", text=result)

    def supports_vision(self) -> bool:
        return False

    def supports_thinking(self) -> bool:
        return False

    def supports_native_tools(self) -> bool:
        return False