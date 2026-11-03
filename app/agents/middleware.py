"""Middleware shared by the main agent and the `visualizer` subagent (see app/agents/agent.py)."""

from deepagents import FilesystemMiddleware
from deepagents.backends import LangSmithSandbox
from deepagents.middleware import SkillsMiddleware, SummarizationMiddleware
from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware, PIIMiddleware
from langchain_core.language_models import BaseChatModel

from app.agents.custom_middleware import InjectionGuardMiddleware
from app.agents.llm import build_fallback_model
from app.config.settings import settings
from app.sandbox.sandboxbackend import SANDBOX_SKILLS_PATH

# Replaces deepagents' generic skills prompt (~700 tokens, with a web-research example)
SKILLS_PROMPT = """## Skills
{skills_locations}{skills_load_warnings}
{skills_list}
Read a skill's SKILL.md with `read_file` (limit=1000) only when the task needs it.
"""

# Personal ID numbers in free text, redacted as [REDACTED_ID]. Only formats with hyphens or letters:
# a bare number (a 12-digit total, a UPI reference) is never hidden. Whole ID columns are
# pseudonymized at upload instead (_pseudonymize_id_columns in app/api/endpoints.py).
ID_PATTERN = "|".join([
    r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN: 123-45-6789
    r"\b[2-9]\d{3}-\d{4}-\d{4}\b",  # Aadhaar written with hyphens: 2345-6789-0123
    r"\b[A-Z]{5}\d{4}[A-Z]\b",  # PAN: ABCDE1234F
])


def sandbox_middleware(model: BaseChatModel, backend: LangSmithSandbox) -> list[AgentMiddleware]:
    """File + `execute` tools, summarization and skills - for the main agent and the subagent."""
    fallback = build_fallback_model()
    return [
        # A failed model call (overloaded, rate limited, timed out) is retried on LLM_FALLBACK. First in
        # the list, so the retry goes through all the middleware below again. Summaries are still
        # written by the main model only.
        *([ModelFallbackMiddleware(fallback)] if fallback else []),
        # Emails and ID numbers in the question and tool results (file contents, script output) are
        # replaced with [REDACTED_EMAIL] / [REDACTED_ID] before the model sees them. First in the
        # list, so the redaction happens before summarization reads the messages.
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_tool_results=True),
        PIIMiddleware("id", detector=ID_PATTERN, strategy="redact", apply_to_input=True, apply_to_tool_results=True),
        FilesystemMiddleware(
            backend=backend,
            # grep, glob and delete are left out: never needed here, and their schemas cost ~1,100 tokens
            tools=["ls", "read_file", "write_file", "edit_file", "execute"],
            # Longer script output is saved to a file and only a preview goes to the model
            tool_token_limit_before_evict=settings.tool_output_max_tokens,
        ),
        # Keeps each request under CONTEXT_MAX_TOKENS by summarizing older turns
        SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=("tokens", settings.context_max_tokens),
            keep=("tokens", settings.context_max_tokens // 3),
            trim_tokens_to_summarize=2000,
            # Scripts in older write_file/edit_file calls are cut down - the files stay in the sandbox
            truncate_args_settings={
                "trigger": ("messages", 8),
                "keep": ("messages", 4),
                "max_length": 300,
                "truncation_text": "...(truncated, see the file in the sandbox)",
            },
        ),
        # Only skill names + descriptions in the prompt; SKILL.md is read on demand
        SkillsMiddleware(backend=backend, sources=[SANDBOX_SKILLS_PATH], system_prompt=SKILLS_PROMPT),
        # Strips "ignore previous instructions"-style text from the question and tool results
        InjectionGuardMiddleware(),
    ]
