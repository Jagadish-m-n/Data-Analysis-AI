"""The data-analysis agent. It runs its Python code inside a LangSmith sandbox.

Use it like this (together with app/sandbox/sandboxbackend.py):

    backend = await create_sandbox()
    try:
        await upload_skills(backend)                                         # -> /skills/
        paths = await upload_data_files(backend, [("sales.csv", csv_bytes)])  # -> /data/sales.csv
        agent = build_agent(backend)
        result = await arun_agent(agent, "Which region sold the most?", "abc123", data_files=paths)
    finally:
        await delete_sandbox(backend)

The model comes from LLM_PROVIDER (groq or gemini, see app/agents/llm.py) and the system prompts
from app/agents/prompts.py. Errors are raised as AppError:
    503 AGENT_DISABLED    ENABLE_AI_AGENT=false
    500 MISSING_API_KEY   GROQ_API_KEY / GEMINI_API_KEY is not set
    400 BAD_REQUEST       the question is empty
    422 MAX_ITERATIONS    the agent hit MAX_AGENT_ITERATIONS before finishing
    500 AGENT_ERROR       anything else (the cause is in the server logs)
"""

from typing import Any

from deepagents import SubAgent, SubAgentMiddleware
from deepagents.backends import LangSmithSandbox
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.agents.llm import build_model
from app.agents.middleware import sandbox_middleware
from app.agents.prompts import SYSTEM_PROMPT, VISUALIZER_PROMPT
from app.config.settings import settings
from app.exception.custom_exception import AppError
from app.logger.custom_logger import get_logger
from app.sandbox.sandboxbackend import SANDBOX_OUTPUT_PATH

logger = get_logger(__name__)


def build_agent(backend: LangSmithSandbox) -> CompiledStateGraph:
    """The main agent plus a `visualizer` subagent for charts, both working in this sandbox."""
    if not settings.enable_ai_agent:
        raise AppError(503, "The AI agent is disabled (ENABLE_AI_AGENT=false).", code="AGENT_DISABLED")
    model = build_model()

    visualizer: SubAgent = {
        "name": "visualizer",
        "description": (
            "Draws charts from a prepared CSV and saves them as PNG files "
            f"in {SANDBOX_OUTPUT_PATH}. Give it the prepared CSV with the exact numbers to plot "
            "(already cleaned and aggregated), the chart type, and the output name."
        ),
        "system_prompt": VISUALIZER_PROMPT,
        "tools": [],
        "model": model,
        # Subagents get no default middleware, so it needs its own file/execute tools and skills
        "middleware": sandbox_middleware(model, backend),
    }

    return create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            *sandbox_middleware(model, backend),
            # `task` tool for delegating to the visualizer subagent
            SubAgentMiddleware(backend=backend, subagents=[visualizer]),
        ],
        # Keeps the conversation between questions, so follow-ups work
        checkpointer=InMemorySaver(),
        name="data-analysis-agent",
    )


async def arun_agent(
    agent: CompiledStateGraph, message: str, thread_id: str, data_files: list[str] | None = None
) -> dict[str, Any]:
    """Send one question to the agent and return its final state (with "messages").

    data_files are the sandbox paths of the uploaded files. They are added to the message so
    the user never has to type a path.
    """
    if not message.strip():
        raise AppError(400, "The question for the AI agent is empty.")
    if data_files:
        message = f"{message}\n\n(Uploaded data files: {', '.join(data_files)})"

    config = {
        "configurable": {"thread_id": thread_id},
        # Each model call + tool call is roughly 2 graph steps
        "recursion_limit": settings.max_agent_iterations * 2,
    }
    try:
        return await agent.ainvoke({"messages": [{"role": "user", "content": message}]}, config=config)
    except GraphRecursionError as e:
        raise AppError(
            422,
            f"The agent stopped after {settings.max_agent_iterations} steps (MAX_AGENT_ITERATIONS) "
            "without finishing. Ask a simpler question or raise the limit.",
            code="MAX_ITERATIONS",
        ) from e
    except AppError:
        raise
    except Exception as e:  # LLM or sandbox failure. Cancels are not an Exception, so they pass through
        logger.error("agent_error", thread_id=thread_id, cause=type(e).__name__, cause_message=str(e), exc_info=e)
        raise AppError(500, "The AI agent failed. See the server logs for details.", code="AGENT_ERROR") from e