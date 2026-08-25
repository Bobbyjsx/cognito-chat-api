"""Request Analyzer components for classifying user requests into routing signals."""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from re import Pattern
from typing import Any, ClassVar

from app.ai.router.exceptions import AnalyzerError
from app.ai.router.schemas import RequestAnalysis, RequestContext, TaskType

logger = logging.getLogger(__name__)

ANALYZER_SYSTEM_INSTRUCTION = (
    "You are a high-speed, compact request analysis engine for an AI model router. "
    "Your ONLY job is to analyze the user's input and return a strict JSON object characterizing "
    "the task type, complexity, reasoning depth, coding requirements, creative requirements, "
    "and required capabilities.\n"
    "DO NOT answer the user's prompt. DO NOT solve the task. DO NOT generate conversational text.\n"
    "Respond ONLY with a valid JSON object adhering strictly to the schema."
)

ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": [t.value for t in TaskType],
            "description": "Primary task category",
        },
        "complexity": {
            "type": "number",
            "description": "Overall complexity from 0.0 (trivial) to 1.0 (extreme/complex)",
        },
        "reasoning_required": {
            "type": "number",
            "description": "Need for deep logical/mathematical reasoning from 0.0 to 1.0",
        },
        "context_required": {
            "type": "number",
            "description": "Need for large context or document retention from 0.0 to 1.0",
        },
        "coding_required": {
            "type": "number",
            "description": "Need for coding, syntax, architecture, or debugging from 0.0 to 1.0",
        },
        "creative_required": {
            "type": "number",
            "description": "Need for open-ended creative storytelling or style from 0.0 to 1.0",
        },
        "vision_required": {
            "type": "boolean",
            "description": "Whether image/diagram perception is needed",
        },
        "web_required": {
            "type": "boolean",
            "description": "Whether real-time web grounding/search is needed",
        },
        "tool_calling_required": {
            "type": "boolean",
            "description": "Whether tool/function calling is needed",
        },
        "structured_output_required": {
            "type": "boolean",
            "description": "Whether rigid JSON/data structuring is needed",
        },
    },
    "required": [
        "task_type",
        "complexity",
        "reasoning_required",
        "context_required",
        "coding_required",
        "creative_required",
        "vision_required",
        "web_required",
        "tool_calling_required",
        "structured_output_required",
    ],
}


class BaseRequestAnalyzer(ABC):
    """Abstract interface for request analysis."""

    @abstractmethod
    async def analyze(
        self,
        message: str,
        context: RequestContext | None = None,
    ) -> RequestAnalysis:
        """Analyze a user message and optional context to produce routing signals."""


class HeuristicFallbackAnalyzer(BaseRequestAnalyzer):
    """Deterministic, fast heuristic analyzer used as a baseline and reliable fallback."""

    # Coding patterns
    _CODE_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"(```|\b(def|class|function|const|let|var|import|export|return|async|await|SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|public|private|static|void|int|float|str|interface|type)\b|[{}();<>=]{3,})",
        re.IGNORECASE,
    )
    _CODING_KEYWORDS: ClassVar[set[str]] = {
        "code",
        "refactor",
        "bug",
        "debug",
        "function",
        "algorithm",
        "python",
        "javascript",
        "typescript",
        "react",
        "fastapi",
        "sql",
        "html",
        "css",
        "api",
        "endpoint",
        "regex",
        "git",
        "docker",
        "deploy",
        "script",
        "unit test",
        "mock",
        "patch",
        "traceback",
    }

    # Reasoning / Math patterns
    _MATH_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"(\b(solve|calculate|equation|integral|derivative|matrix|probability|proof|theorem|theorem)\b|[\d+\-*/^=()]{6,})",
        re.IGNORECASE,
    )
    _REASONING_KEYWORDS: ClassVar[set[str]] = {
        "why",
        "how come",
        "prove",
        "deduce",
        "logic",
        "analyze",
        "evaluate",
        "compare",
        "tradeoff",
        "investigate",
        "diagnose",
        "inconsistency",
    }

    # Creative patterns
    _CREATIVE_KEYWORDS: ClassVar[set[str]] = {
        "write a story",
        "poem",
        "essay",
        "creative",
        "fiction",
        "novel",
        "lyrics",
        "dialogue",
        "roleplay",
        "metaphor",
        "character",
        "plot",
    }

    # Web keywords
    _WEB_KEYWORDS: ClassVar[set[str]] = {
        "latest",
        "current news",
        "today's news",
        "current weather",
        "who is the current",
        "who is ",
        "stock price",
        "recent news",
        "search the web",
        "google search",
        "search google",
        "search for",
        "web search",
        "look up",
        "find online",
        "browse the web",
    }

    # Summarization keywords
    _SUMMARIZATION_KEYWORDS: ClassVar[set[str]] = {
        "summarize",
        "tl;dr",
        "summary",
        "key points",
        "bullet points",
        "synopsis",
        "recap",
    }

    async def analyze(
        self,
        message: str,
        context: RequestContext | None = None,
    ) -> RequestAnalysis:
        text = message.strip()
        lower_text = text.lower()
        word_count = len(text.split())

        has_attachments = context.has_attachments if context else False
        attachment_types = context.attachment_types if context else []
        approx_tokens = context.approximate_context_tokens if context else word_count * 2

        # 1. Vision requirement
        vision_required = has_attachments and any(t in ("image", "pdf") for t in attachment_types)

        # 2. Coding detection
        is_coding = bool(self._CODE_PATTERN.search(text)) or any(kw in lower_text for kw in self._CODING_KEYWORDS)
        coding_score = 0.0
        if is_coding:
            coding_score = 0.9 if "```" in text or "def " in text or "function " in text else 0.7

        # 3. Math & Reasoning detection
        is_math = bool(self._MATH_PATTERN.search(text))
        reasoning_score = 0.2
        if is_math:
            reasoning_score = 0.85
        elif any(kw in lower_text for kw in self._REASONING_KEYWORDS):
            reasoning_score = 0.75
        elif is_coding:
            reasoning_score = 0.65

        # 4. Creative detection
        is_creative = any(kw in lower_text for kw in self._CREATIVE_KEYWORDS)
        creative_score = 0.85 if is_creative else 0.05

        # 5. Summarization detection
        is_summary = any(kw in lower_text for kw in self._SUMMARIZATION_KEYWORDS)

        # 6. Web requirement
        web_required = any(kw in lower_text for kw in self._WEB_KEYWORDS)

        # 7. Context requirement
        context_score = min(1.0, max(0.1, approx_tokens / 100_000))
        if has_attachments:
            context_score = max(context_score, 0.6)

        # 8. Complexity calculation
        base_complexity = 0.25  # default simple prompt
        if word_count > 200:
            base_complexity += 0.2
        elif word_count > 60:
            base_complexity += 0.1

        if is_coding:
            base_complexity = max(
                base_complexity, 0.70 if "refactor" in lower_text or "architecture" in lower_text else 0.55
            )
        if is_math:
            base_complexity = max(base_complexity, 0.75)
        if "analyze" in lower_text or "inconsistency" in lower_text or "compare" in lower_text:
            base_complexity = max(base_complexity, 0.65)
        if has_attachments:
            base_complexity = max(base_complexity, 0.60)

        complexity = min(1.0, max(0.1, base_complexity))

        # 9. Task type assignment
        is_greeting = word_count < 15 and (
            any(w.strip(".,!?") in ("hello", "hi", "hey", "sup", "howdy") for w in lower_text.split())
            or any(p in lower_text for p in ("how are you", "good morning", "good evening", "what's up"))
        )

        if is_coding:
            task_type = TaskType.CODING
        elif is_math:
            task_type = TaskType.MATH_REASONING
        elif is_summary:
            task_type = TaskType.SUMMARIZATION
        elif is_creative:
            task_type = TaskType.CREATIVE_WRITING
        elif "analyze" in lower_text or "investigate" in lower_text:
            task_type = TaskType.ANALYSIS
        elif web_required:
            task_type = TaskType.GENERAL_KNOWLEDGE
        elif is_greeting:
            task_type = TaskType.CONVERSATION
            complexity = 0.15
            reasoning_score = 0.1
            web_required = False
        else:
            task_type = TaskType.GENERAL_KNOWLEDGE

        return RequestAnalysis(
            task_type=task_type,
            complexity=round(complexity, 2),
            reasoning_required=round(reasoning_score, 2),
            context_required=round(context_score, 2),
            coding_required=round(coding_score, 2),
            creative_required=round(creative_score, 2),
            vision_required=vision_required,
            web_required=web_required,
            tool_calling_required=is_coding or web_required,
            structured_output_required=False,
            confidence=0.85,
        )


def _extract_text_from_genai_response(response: Any) -> str:
    """Safely extract text from a google-genai response object.

    Handles cases where thinking tokens or thought signatures are attached to parts.
    """
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        if content:
            parts = getattr(content, "parts", None) or []
            texts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text and isinstance(part_text, str):
                    texts.append(part_text)
            if texts:
                return "".join(texts)

    try:
        text = getattr(response, "text", None)
        if text and isinstance(text, str):
            return text
    except Exception as exc:
        logger.debug("Failed to read response.text attribute: %s", exc)

    return ""


class GeminiFlashLiteAnalyzer(BaseRequestAnalyzer):
    """LLM-based classifier using Gemini Flash-Lite (fast & structured JSON output)."""

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        api_key: str | None = None,
        timeout_seconds: float = 3.0,
    ):
        self.model_name = model_name
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os

            from google import genai

            from app.core.config import settings

            api_key = self._api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._client

    def _build_prompt(self, message: str, context: RequestContext | None = None) -> str:
        # Keep prompt concise to minimize latency and token overhead
        meta_lines = []
        if context:
            if context.conversation_message_count > 0:
                meta_lines.append(f"Conversation turns so far: {context.conversation_message_count}")
            if context.approximate_context_tokens > 0:
                meta_lines.append(f"Estimated context tokens: {context.approximate_context_tokens}")
            if context.has_attachments:
                attachment_strs = [t.value if hasattr(t, "value") else str(t) for t in context.attachment_types]
                meta_lines.append(f"Attachments present: {', '.join(attachment_strs) or 'yes'}")
            if context.recent_message_summary:
                meta_lines.append(f"Recent context: {context.recent_message_summary[:200]}")

        context_str = "\n[Context Metadata]\n" + "\n".join(meta_lines) if meta_lines else ""

        # Truncate user prompt if excessively long for routing analysis
        truncated_msg = message.strip()
        if len(truncated_msg) > 1500:
            truncated_msg = truncated_msg[:1500] + "... [truncated for classification]"

        return f'{context_str}\n[User Prompt to Classify]:\n"{truncated_msg}"\n\nJSON Output:'

    async def analyze(
        self,
        message: str,
        context: RequestContext | None = None,
    ) -> RequestAnalysis:
        if not message or not message.strip():
            return RequestAnalysis(
                task_type=TaskType.CONVERSATION,
                complexity=0.1,
                reasoning_required=0.0,
                context_required=0.0,
                coding_required=0.0,
                creative_required=0.0,
                vision_required=False,
                web_required=False,
                tool_calling_required=False,
                structured_output_required=False,
                confidence=1.0,
            )

        client = self._get_client()
        prompt = self._build_prompt(message, context)

        from google.genai import types

        try:
            config = types.GenerateContentConfig(
                system_instruction=ANALYZER_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ANALYSIS_JSON_SCHEMA,
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            import asyncio

            analysis_start = time.perf_counter()
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - analysis_start) * 1000.0

            text = _extract_text_from_genai_response(response)
            if not text:
                raise AnalyzerError("Flash-Lite analyzer returned empty content.")

            clean_text = text.strip()
            if clean_text.startswith("```"):
                lines = clean_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()

            data = json.loads(clean_text)
            analysis = RequestAnalysis.model_validate(data)

            # Reconcile with explicit context if attachments present
            if context and context.has_attachments and any(t in ("image", "pdf") for t in context.attachment_types):
                analysis.vision_required = True

            logger.info(
                "[SmartRouter][Analyzer] Gemini Flash-Lite response (%.1f ms): %s | Parsed analysis: task_type=%s, complexity=%.2f, reasoning=%.2f, coding=%.2f",
                latency_ms,
                clean_text.replace("\n", " "),
                analysis.task_type.value,
                analysis.complexity,
                analysis.reasoning_required,
                analysis.coding_required,
            )
            return analysis

        except Exception as exc:
            err_type = type(exc).__name__
            err_msg = getattr(exc, "message", None) or str(exc) or repr(exc)
            logger.warning(
                "[SmartRouter][Analyzer] Gemini Flash-Lite (%s) classification failed: [%s] %s. Triggering fallback.",
                self.model_name,
                err_type,
                err_msg,
            )
            raise AnalyzerError(f"Flash-Lite analysis failed ({err_type}: {err_msg})") from exc


class CompositeRequestAnalyzer(BaseRequestAnalyzer):
    """Resilient analyzer: runs fast heuristics on simple prompts, and Flash-Lite on complex requests."""

    def __init__(
        self,
        primary_analyzer: BaseRequestAnalyzer | None = None,
        fallback_analyzer: BaseRequestAnalyzer | None = None,
    ):
        self.primary_analyzer = primary_analyzer
        self.fallback_analyzer = fallback_analyzer or HeuristicFallbackAnalyzer()

    async def analyze(
        self,
        message: str,
        context: RequestContext | None = None,
    ) -> RequestAnalysis:
        words = message.strip().split()
        is_short_prompt = len(words) <= 12
        has_complex_cues = any(
            kw in message.lower() for kw in ("def ", "class ", "import ", "select ", "```", "{", "}")
        )
        has_attachments = context.has_attachments if context else False
        is_multi_turn = (context.conversation_message_count > 3) if context else False

        # Fast-path for short single-sentence definitions, queries, and conversational messages
        if is_short_prompt and not has_complex_cues and not has_attachments and not is_multi_turn:
            start = time.perf_counter()
            fast_res = await self.fallback_analyzer.analyze(message, context)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "[SmartRouter][Analyzer] Fast-path heuristic analysis completed in %.2f ms | task_type=%s, complexity=%.2f, reasoning=%.2f",
                latency_ms,
                fast_res.task_type.value,
                fast_res.complexity,
                fast_res.reasoning_required,
            )
            return fast_res

        if self.primary_analyzer:
            try:
                result = await self.primary_analyzer.analyze(message, context)
                return result
            except Exception as exc:
                err_type = type(exc).__name__
                err_msg = getattr(exc, "message", None) or str(exc) or repr(exc)
                logger.warning(
                    "[SmartRouter][Analyzer] Primary analyzer failed (%s: %s). Falling back to deterministic heuristic analyzer.",
                    err_type,
                    err_msg,
                )

        start = time.perf_counter()
        fallback_res = await self.fallback_analyzer.analyze(message, context)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[SmartRouter][Analyzer] Heuristic fallback analysis (%.1f ms): task_type=%s, complexity=%.2f, reasoning=%.2f, coding=%.2f, vision=%s",
            latency_ms,
            fallback_res.task_type.value,
            fallback_res.complexity,
            fallback_res.reasoning_required,
            fallback_res.coding_required,
            fallback_res.vision_required,
        )
        return fallback_res
