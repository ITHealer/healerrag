"""Generated answer quality judgment for Agentic RAG."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.services.agentic.models import AgenticRetrievedChunk, ExecutionPlan, ResponseJudgment
from app.services.agentic.prompts import RESPONSE_JUDGE_SYSTEM_PROMPT
from app.services.llm.types import LLMMessage, LLMResult

logger = logging.getLogger(__name__)


class ResponseJudge:
    """Judge faithfulness, completeness, and plan coverage for an answer."""

    def __init__(
        self,
        *,
        llm_provider: Any | None = None,
        faithfulness_threshold: float = 0.7,
        completeness_threshold: float = 0.7,
        timeout_seconds: float = 5.0,
        max_tokens: int = 1200,
        max_evidence_items: int = 10,
        use_llm: bool = True,
    ) -> None:
        self._llm_provider = llm_provider
        self._faithfulness_threshold = min(1.0, max(0.0, float(faithfulness_threshold)))
        self._completeness_threshold = min(1.0, max(0.0, float(completeness_threshold)))
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_tokens = max(256, int(max_tokens))
        self._max_evidence_items = max(1, int(max_evidence_items))
        self._use_llm = use_llm

    @classmethod
    def from_settings(
        cls,
        *,
        llm_provider: Any | None = None,
        config: Any | None = None,
    ) -> "ResponseJudge":
        """Build response judge from app settings without import-time side effects."""

        if config is None:
            from app.core.config import settings as app_settings

            config = app_settings

        return cls(
            llm_provider=llm_provider,
            faithfulness_threshold=config.AGENTIC_RAG_FAITHFULNESS_THRESHOLD,
            completeness_threshold=config.AGENTIC_RAG_COMPLETENESS_THRESHOLD,
            timeout_seconds=config.AGENTIC_JUDGE_TIMEOUT,
            max_evidence_items=config.AGENTIC_MAX_FINAL_CHUNKS,
            use_llm=True,
        )

    async def judge(
        self,
        *,
        original_query: str,
        generated_answer: str,
        chunks: list[AgenticRetrievedChunk],
        execution_plan: ExecutionPlan,
    ) -> ResponseJudgment:
        """Return structured quality judgment for a generated answer."""

        if not generated_answer.strip():
            return ResponseJudgment(
                pass_judge=False,
                faithfulness_score=0.0,
                completeness_score=0.0,
                issues=["Generated answer is empty."],
                reasoning="Empty answers cannot satisfy the request.",
            )

        if self._use_llm:
            try:
                judgment = await asyncio.wait_for(
                    self._judge_with_llm(
                        original_query=original_query,
                        generated_answer=generated_answer,
                        chunks=chunks,
                        execution_plan=execution_plan,
                    ),
                    timeout=self._timeout_seconds,
                )
                return self._normalize_judgment(judgment)
            except Exception as exc:
                logger.info("ResponseJudge fallback used: %s", exc)

        return self._rule_based_judgment(
            generated_answer=generated_answer,
            chunks=chunks,
            execution_plan=execution_plan,
        )

    async def _judge_with_llm(
        self,
        *,
        original_query: str,
        generated_answer: str,
        chunks: list[AgenticRetrievedChunk],
        execution_plan: ExecutionPlan,
    ) -> ResponseJudgment:
        provider = self._llm_provider
        if provider is None:
            from app.services.llm import get_llm_provider

            provider = get_llm_provider()
            self._llm_provider = provider

        user_content = (
            "Return only JSON. Do not wrap it in markdown.\n\n"
            f"Faithfulness threshold: {self._faithfulness_threshold}\n"
            f"Completeness threshold: {self._completeness_threshold}\n\n"
            f"Original query:\n{original_query}\n\n"
            f"Execution plan JSON:\n{execution_plan.model_dump_json()}\n\n"
            f"Generated answer:\n{generated_answer}\n\n"
            f"Selected evidence:\n{self._evidence_preview(chunks)}"
        )
        result = await provider.acomplete(
            [LLMMessage(role="user", content=user_content)],
            system_prompt=RESPONSE_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        text = result.content if isinstance(result, LLMResult) else str(result)
        data = self._extract_json_object(text)
        data = self._coerce_judgment_data(data)
        return ResponseJudgment.model_validate(data)

    @staticmethod
    def _coerce_judgment_data(data: dict[str, Any]) -> dict[str, Any]:
        coerced = dict(data)
        if "faithfulness_score" not in coerced and "faithfulness" in coerced:
            coerced["faithfulness_score"] = coerced["faithfulness"]
        if "completeness_score" not in coerced and "completeness" in coerced:
            coerced["completeness_score"] = coerced["completeness"]
        if "pass_judge" not in coerced:
            raw_status = str(
                coerced.get("status")
                or coerced.get("overall")
                or coerced.get("verdict")
                or ""
            ).strip().lower()
            if raw_status in {"pass", "passed", "accept", "accepted", "ok"}:
                coerced["pass_judge"] = True
            elif raw_status in {"fail", "failed", "reject", "rejected"}:
                coerced["pass_judge"] = False
            else:
                faithfulness = float(coerced.get("faithfulness_score", 0.0) or 0.0)
                completeness = float(coerced.get("completeness_score", 0.0) or 0.0)
                coerced["pass_judge"] = faithfulness >= 0.7 and completeness >= 0.7
        issues = coerced.get("issues")
        if isinstance(issues, str):
            coerced["issues"] = [issues]
        elif not isinstance(issues, list):
            coerced["issues"] = []
        coerced.setdefault("reasoning", str(coerced.get("reason") or coerced.get("explanation") or ""))
        return coerced

    def _normalize_judgment(self, judgment: ResponseJudgment) -> ResponseJudgment:
        issues = list(judgment.issues)
        pass_judge = judgment.pass_judge
        if judgment.faithfulness_score < self._faithfulness_threshold:
            pass_judge = False
            issues.append("Faithfulness score is below threshold.")
        if judgment.completeness_score < self._completeness_threshold:
            pass_judge = False
            issues.append("Completeness score is below threshold.")

        return judgment.model_copy(
            update={
                "pass_judge": pass_judge,
                "issues": self._dedupe(issues),
            }
        )

    def _rule_based_judgment(
        self,
        *,
        generated_answer: str,
        chunks: list[AgenticRetrievedChunk],
        execution_plan: ExecutionPlan,
    ) -> ResponseJudgment:
        answer_text = generated_answer.casefold()
        evidence_text = "\n".join(chunk.content for chunk in chunks).casefold()
        issues: list[str] = []

        faithfulness_score = self._faithfulness_score(answer_text=answer_text, evidence_text=evidence_text)
        if faithfulness_score < self._faithfulness_threshold:
            issues.append("Answer contains terms that are not well supported by selected evidence.")

        required_items = self._required_item_descriptions(execution_plan)
        covered_items = [item for item in required_items if self._aspect_is_covered(item, answer_text)]
        completeness_score = len(covered_items) / max(1, len(required_items)) if required_items else 0.8
        if required_items and completeness_score < self._completeness_threshold:
            missing_items = [item for item in required_items if item not in covered_items]
            issues.extend(f"Missing planned item: {item}" for item in missing_items)

        pass_judge = (
            faithfulness_score >= self._faithfulness_threshold
            and completeness_score >= self._completeness_threshold
        )
        return ResponseJudgment(
            pass_judge=pass_judge,
            faithfulness_score=round(faithfulness_score, 4),
            completeness_score=round(completeness_score, 4),
            issues=self._dedupe(issues),
            reasoning="Rule-based judge compared answer terms and batch_now coverage against selected evidence.",
        )

    def _evidence_preview(self, chunks: list[AgenticRetrievedChunk]) -> str:
        preview_items: list[str] = []
        for index, chunk in enumerate(
            sorted(chunks, key=lambda item: (-item.score, item.chunk_id))[: self._max_evidence_items],
            start=1,
        ):
            content = re.sub(r"\s+", " ", chunk.content).strip()[:900]
            preview_items.append(
                f"[{index}] chunk_id={chunk.chunk_id} source={chunk.source.value} score={chunk.score:.4f}\n"
                f"{content}"
            )
        if len(chunks) > self._max_evidence_items:
            preview_items.append(f"... {len(chunks) - self._max_evidence_items} additional chunks omitted")
        return "\n\n".join(preview_items)

    @staticmethod
    def _required_item_descriptions(execution_plan: ExecutionPlan) -> list[str]:
        item_by_id = {item.item_id: item for item in execution_plan.items}
        return [
            item_by_id[item_id].description
            for item_id in execution_plan.batch_now
            if item_id in item_by_id and item_by_id[item_id].description.strip()
        ]

    @staticmethod
    def _faithfulness_score(*, answer_text: str, evidence_text: str) -> float:
        if not evidence_text.strip():
            return 0.0

        answer_tokens = ResponseJudge._content_tokens(answer_text)
        if not answer_tokens:
            return 0.8

        evidence_tokens = set(ResponseJudge._content_tokens(evidence_text))
        supported = sum(1 for token in set(answer_tokens) if token in evidence_tokens)
        return supported / max(1, len(set(answer_tokens)))

    @staticmethod
    def _aspect_is_covered(aspect: str, text: str) -> bool:
        tokens = [
            token
            for token in ResponseJudge._content_tokens(aspect)
            if token not in {"analyze", "analysis", "explain", "describe"}
        ]
        if not tokens:
            return True
        required_matches = 1 if len(tokens) == 1 else max(2, int(len(set(tokens)) * 0.75 + 0.999))
        matches = sum(1 for token in set(tokens) if token in text)
        return matches >= required_matches

    @staticmethod
    def _content_tokens(text: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "using",
            "into",
            "about",
            "answer",
            "based",
            "evidence",
            "source",
            "chunk",
        }
        return [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9_.$-]{3,}", text)
            if token.casefold() not in stopwords
        ]

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON output is not an object")
        return parsed


__all__ = ["ResponseJudge"]
