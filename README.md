# Data Analysis AI

Upload CSV files, ask a question in plain English, and get an answer backed by real computation,
plus charts and a Markdown report.

An AI agent (LangChain + [deepagents](https://github.com/langchain-ai/deepagents), running on
Groq or Gemini) writes Python scripts and runs them in an isolated
[LangSmith sandbox](https://docs.smith.langchain.com/), so the generated code never runs on the
server itself. The agent is told to compute every number in its answer with a script and never
to estimate it.

## Features

- **Web UI**: upload files, chat with follow-up questions, watch the agent's steps live, and
  browse the files in the sandbox (`/data`, `/output`, `/skills`).
- **REST API**: a one-shot `POST /api/agent/analyze` for scripts and other services. It returns
  the answer and the output files base64-encoded.
- **Charts and reports**: a `visualizer` subagent draws charts with matplotlib/seaborn. The
  agent saves `.png`, `.svg`, `.md` and `.csv` files in `/output`.
- **Skills**: short how-to guides in [app/skills/](app/skills/) (loading, cleaning, analysis,
  visualization, reporting). The agent reads one only when the task needs it.
- **Two LLM providers**: Groq or Gemini, with an optional fallback model when the main one
  fails.
- **Privacy and safety**:
  - ID columns (`customer_id`, `ssn`, `PAN`, `aadhaar_no`, ...) are replaced with pseudonyms at
    upload.
  - Emails and ID numbers are redacted before the model sees them.
  - "Ignore previous instructions"-style text in the data is stripped out.
- **Limits**: optional API key, a rate limit per client IP, upload size and count limits, and a
  cap on agent steps.

## How it works

```
Browser / API client
        │  CSV files + question
        ▼
FastAPI server (this repo)
        │  1. starts a LangSmith sandbox
        │  2. uploads app/skills/ → /skills/ and the CSV files → /data/
        │  3. runs the agent: it writes Python scripts and executes them in the sandbox
        │  4. downloads charts and reports from /output/
        │  5. deletes the sandbox
        ▼
Answer + files
```

In the web UI, each conversation keeps its sandbox between questions. The sandbox is deleted when
you end the conversation, or after 30 minutes of inactivity.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/), or Docker
- An API key for the LLM: [Groq](https://console.groq.com/keys) or
  [Gemini](https://aistudio.google.com/apikey)
- A [LangSmith](https://smith.langchain.com/) API key. The sandboxes run on LangSmith.

## Quick start

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` in the project root (`.env` is git-ignored, so your keys
   stay out of the repository):

   ```bash
   cp .env.example .env              # macOS / Linux
   Copy-Item .env.example .env       # Windows PowerShell
   ```

   Then open `.env` and fill in at least your LLM key and your LangSmith key:

   ```ini
   LLM_PROVIDER=groq
   GROQ_API_KEY=your-groq-key
   LANGSMITH_API_KEY=your-langsmith-key
   ```

   All the other settings are listed under [Configuration](#configuration).

3. (Recommended) Build the sandbox snapshot, so pandas, numpy, matplotlib, seaborn and scipy
   are already installed in every new sandbox:

   ```bash
   uv run python -m scripts.build_snapshot
   ```

   The script prints the snapshot id. If `SANDBOX_SNAPSHOT` is empty in `.env`, it also writes
   the id there. Without a snapshot, sandboxes boot from LangSmith's default image.

4. Start the server:

   ```bash
   uv run main.py
   ```

   - Web UI: <http://127.0.0.1:8000/>
   - API docs (Swagger): <http://127.0.0.1:8000/docs>

## Run with Docker

The [Dockerfile](Dockerfile) builds an image with the locked dependencies from `uv.lock` and
runs the server as a non-root user. `.env` is not copied into the image: pass your settings when
the container starts.

```bash
docker build -t data-analysis-ai .

docker run -d --name data-analysis-ai \
  -p 8000:8000 \
  --env-file .env \
  -v data-analysis-logs:/app/logs \
  data-analysis-ai
```

Then open <http://localhost:8000/>.

| Task                    | Command                                       |
| ----------------------- | --------------------------------------------- |
| Follow the logs (JSON)  | `docker logs -f data-analysis-ai`             |
| Check health            | `docker inspect --format '{{.State.Health.Status}}' data-analysis-ai` |
| Stop                    | `docker stop -t 60 data-analysis-ai`          |

Notes:

- **Stop with `-t 60`.** On shutdown the server deletes every open sandbox. Docker's default
  10 seconds can kill it before that finishes. Sandboxes left behind are deleted by LangSmith
  after 30 idle minutes.
- **One container, one worker.** Conversations, sandboxes and the rate limit are kept in memory.
  Don't add `--workers`, and don't put several replicas behind a load balancer without sticky
  sessions.
- **Set `API_AUTH_KEY`** before the port is reachable by anyone else. Without it, the API needs
  no login.
- `--env-file` reads values literally: write `KEY=value`, without quotes or trailing `# comments`.
- Build the sandbox snapshot from your machine (see [Quick start](#quick-start)), not from the
  container.

## Configuration

Every setting is an environment variable, or a line in `.env`. Only the LLM key and
`LANGSMITH_API_KEY` are required. See [app/config/settings.py](app/config/settings.py).

### LLM

| Variable                | Default               | Description                                                       |
| ----------------------- | --------------------- | ----------------------------------------------------------------- |
| `LLM_PROVIDER`          | `groq`                | `groq` or `gemini`                                                |
| `LLM_FALLBACK`          | *(empty)*             | Model used when the main one fails, as `provider:model`, e.g. `gemini:gemini-3.5-flash-lite` |
| `GROQ_API_KEY`          | *(empty)*             | Required when `LLM_PROVIDER=groq`                                 |
| `GROQ_MODEL`            | `openai/gpt-oss-120b` |                                                                   |
| `GROQ_TIMEOUT`          | `90`                  | Seconds one model call may take                                   |
| `GROQ_REASONING_EFFORT` | `low`                 | Reasoning models only (gpt-oss, qwen3). Leave empty for others.   |
| `GEMINI_API_KEY`        | *(empty)*             | Required when `LLM_PROVIDER=gemini`                               |
| `GEMINI_MODEL`          | `gemini-3.8-flash`    |                                                                   |
| `GEMINI_TIMEOUT`        | `90`                  | Seconds one model call may take (at least 10)                     |

### Sandbox and tracing

| Variable                 | Default               | Description                                                  |
| ------------------------ | --------------------- | ------------------------------------------------------------ |
| `LANGSMITH_API_KEY`      | *(empty)*             | Required: runs the sandboxes                                 |
| `SANDBOX_SNAPSHOT`       | *(empty)*             | Snapshot id, `name:tag` or name to boot sandboxes from       |
| `SANDBOX_TRANSPORT`      | `websocket`           | `sse` sends commands over plain HTTPS: use it when a proxy blocks WebSockets |
| `SANDBOX_CREATE_TIMEOUT` | `120`                 | Seconds to wait for a new sandbox                            |
| `LANGSMITH_TRACING`      | `false`               | Send traces of each agent run to LangSmith. The LangSmith SDK reads it from the process environment, not from `.env`: it works with Docker's `--env-file`, or locally with `uv run --env-file .env main.py`. |
| `LANGSMITH_PROJECT`      | `data-analysis-ai`    | LangSmith project for the traces                             |

### Server and limits

| Variable                 | Default              | Description                                                   |
| ------------------------ | -------------------- | ------------------------------------------------------------- |
| `API_AUTH_KEY`           | *(empty)*            | When set, clients must send it in the `X-API-Key` header      |
| `API_HOST` / `API_PORT`  | `127.0.0.1` / `8000` | Where `uv run main.py` listens (the Docker image always uses `0.0.0.0:8000`) |
| `APP_NAME`               | `Data Analysis AI`   |                                                               |
| `APP_ENV`                | `development`        |                                                               |
| `LOG_LEVEL`              | `INFO`               | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`             |
| `ENABLE_AI_AGENT`        | `true`               | `false` rejects every analysis with 503                       |
| `MAX_AGENT_ITERATIONS`   | `40`                 | Model and tool calls before the agent is stopped              |
| `CONTEXT_MAX_TOKENS`     | `5000`               | Older messages are summarized above this size                 |
| `TOOL_OUTPUT_MAX_TOKENS` | `1000`               | Longer script output is saved to a file instead               |
| `MAX_QUESTION_CHARS`     | `2000`               |                                                               |
| `MAX_UPLOAD_FILES`       | `10`                 | Files per upload                                              |
| `MAX_UPLOAD_MB`          | `50`                 | Size limit per file                                           |
| `RATE_LIMIT_PER_MINUTE`  | `5`                  | Questions per client IP per minute. `0` means no limit.       |

## API

The interactive docs are at `/docs`. When `API_AUTH_KEY` is set, every route except
`/api/health` needs the `X-API-Key` header.

### One-shot analysis

```bash
curl -X POST http://localhost:8000/api/agent/analyze \
  -H "X-API-Key: $API_AUTH_KEY" \
  -F "question=Which region had the highest revenue last quarter? Plot revenue by month." \
  -F "files=@sales.csv"
```

```json
{
  "thread_id": "…",
  "status": "completed",
  "answer": "The North region had the highest revenue: …",
  "files": [{ "name": "revenue_by_month.png", "media_type": "image/png", "content_base64": "…" }],
  "warnings": []
}
```

To be able to cancel an analysis, send your own `request_id` form field (8 to 64 letters, digits,
`-` or `_`). Then call `POST /api/agent/cancel/{request_id}` to stop it.

### All routes

| Method   | Route                                 | Purpose                                              |
| -------- | ------------------------------------- | ---------------------------------------------------- |
| `GET`    | `/api/health`                         | Is the server up? (no API key needed)                |
| `GET`    | `/api/status`                         | Configuration summary (never shows the keys)         |
| `POST`   | `/api/agent/analyze`                  | CSV files and a question in, answer and files out    |
| `POST`   | `/api/agent/cancel/{request_id}`      | Stop a running analysis                              |
| `POST`   | `/api/threads`                        | Upload CSV files and start a conversation (web UI)   |
| `GET`    | `/api/threads/{id}`                   | The conversation so far                              |
| `POST`   | `/api/threads/{id}/messages`          | Ask a question. The steps stream back as Server-Sent Events. |
| `POST`   | `/api/threads/{id}/stop`              | Stop the current answer                              |
| `DELETE` | `/api/threads/{id}`                   | End the conversation and delete its sandbox          |
| `GET`    | `/api/threads/{id}/tree`              | Files in `/data`, `/output` and `/skills`            |
| `GET`    | `/api/threads/{id}/file?path=…`       | Raw content of one sandbox file                      |

Errors always have the same JSON shape:
`{"error": {"code": "FILE_TOO_LARGE", "message": "…"}}`.

## Project layout

```
app/
  agents/      the agent, the visualizer subagent, prompts, LLM setup and middleware
  api/         FastAPI app, routes and response models
  config/      settings (environment variables / .env)
  exception/   AppError and the JSON error handlers
  logger/      JSON logging to the console and logs/
  sandbox/     LangSmith sandbox: create, upload, download, delete
  skills/      SKILL.md guides uploaded to every sandbox
  static/      web UI (HTML, CSS, JS)
scripts/
  build_snapshot.py   builds the sandbox snapshot with the data libraries
tests/                pytest tests
main.py               entry point: uv run main.py
Dockerfile
```

## Tests

```bash
uv run pytest -v
```

Most tests run offline. Two tests in `tests/test_api.py` call the LLM provider and LangSmith for
real, to check that your keys work. They are skipped when the keys are not set.

## Logs

Each log line is a JSON object. Logs are printed to the console and also written to
`logs/<start time>.log`. In Docker, that folder is `/app/logs`; the `docker run` command above
keeps it in the `data-analysis-logs` volume.
