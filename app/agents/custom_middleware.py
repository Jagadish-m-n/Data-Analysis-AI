"""Strips prompt-injection text from what the model reads (the question and tool results).

Uploaded files can hold text like "ignore previous instructions" in a cell; the agent sees it
through `read_file` or a script's output. Before every model call the matching text is replaced
with a placeholder, so the model can still report it as a data-quality issue, and a
"possible_injection" warning is logged. The stored conversation is not changed.
"""

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config

from app.logger.custom_logger import get_logger

logger = get_logger(__name__)

PLACEHOLDER = "[removed: possible prompt injection]"

# Phrases that try to give the model new orders. Kept narrow, so normal data text is not hit.
_INJECTION_RE = re.compile(
    r"|".join([
        r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?"
        r"(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules)",
        r"\b(?:reveal|show|print|repeat|output)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+prompt|instructions)",
        r"\bnew\s+instructions\s*:",
        r"\byou\s+are\s+now\s+(?:a|an|in)\b",
        r"\bdo\s+not\s+(?:tell|inform)\s+the\s+user\b",
        r"<\|?/?(?:system|im_start|im_end)\|?>|\[/?INST\]",  # chat-template tokens
    ]),
    re.IGNORECASE,
)


def scan_for_injection(text: str) -> list[str]:
    """The injection phrases found in text (empty list = clean)."""
    return [m.group(0)[:80] for m in _INJECTION_RE.finditer(text)]


def _text_of(content: str | list[Any]) -> str:
    """The text of a message: a plain string, or the text parts of a content-block list."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    )


def _sanitize(content: str | list[Any]) -> str | list[Any]:
    """The same content with every injection phrase replaced by PLACEHOLDER."""
    if isinstance(content, str):
        return _INJECTION_RE.sub(PLACEHOLDER, content)
    return [
        _INJECTION_RE.sub(PLACEHOLDER, block) if isinstance(block, str)
        else {**block, "text": _INJECTION_RE.sub(PLACEHOLDER, block["text"])} if block.get("type") == "text"
        else block
        for block in content
    ]


def _thread_id() -> str | None:
    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:  # not inside an agent run
        return None


class InjectionGuardMiddleware(AgentMiddleware):
    """Sanitizes the question and tool results before each model call."""

    def _guard(self, request):
        # Messages after the last AIMessage are new since the previous model call. Older ones are
        # sanitized again on every call (the state keeps the original), but only logged once.
        last_ai = max((i for i, m in enumerate(request.messages) if isinstance(m, AIMessage)), default=-1)
        new_messages, new_hits, changed = [], [], False
        for i, msg in enumerate(request.messages):
            if isinstance(msg, HumanMessage | ToolMessage):  # the question, file contents, script output
                found = scan_for_injection(_text_of(msg.content))
                if found:
                    changed = True
                    if i > last_ai:
                        new_hits.extend(found)
                    msg = msg.model_copy(update={"content": _sanitize(msg.content)})
            new_messages.append(msg)
        if new_hits:
            logger.warning("possible_injection", thread_id=_thread_id(), count=len(new_hits), removed=new_hits[:3])
        return request.override(messages=new_messages) if changed else request

    def wrap_model_call(self, request, handler):
        return handler(self._guard(request))

    async def awrap_model_call(self, request, handler):
        # The API runs the agent with ainvoke/astream, which only call this async version
        return await handler(self._guard(request))
