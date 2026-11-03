"""Build the sandbox snapshot "data-analysis": the default sandbox image with pandas, numpy,
matplotlib, seaborn and scipy already installed, so the agent does not have to install them.

    uv run python -m scripts.build_snapshot

It starts a temporary sandbox, adds a `python` command (the image only has `python3`),
pip installs the libraries at the versions locked in uv.lock,
checks that they import, captures the sandbox as a snapshot tagged with today's date and prints
the snapshot id. If SANDBOX_SNAPSHOT is missing or empty in .env it writes SANDBOX_SNAPSHOT=<id>
there; a value that is already set is left alone.
The temporary sandbox is always deleted, also when a step fails.

Why capture_snapshot and not create_snapshot_from_dockerfile: the Dockerfile route cannot set a
tag, needs a FROM image (the default sandbox image is not exposed, so it would not match), and
does the same steps anyway - it starts a builder sandbox and captures it - with BuildKit on top.
"""

import tomllib
import uuid
from datetime import date

from langsmith.sandbox import ResourceNotFoundError

from app.config.settings import PROJECT_ROOT, settings
from app.sandbox.sandboxbackend import SANDBOX_IDLE_TTL_SECONDS, client

SNAPSHOT_NAME = "data-analysis"
LIBRARIES = ["pandas", "numpy", "matplotlib", "seaborn", "scipy"]
IMPORT_CHECK = (
    'python -c "import pandas, numpy, matplotlib, seaborn, scipy; '
    'print(pandas.__version__, numpy.__version__, matplotlib.__version__, seaborn.__version__, scipy.__version__)"'
)
STEP_TIMEOUT = 600  # seconds for the install and for the capture
ENV_FILE = PROJECT_ROOT / ".env"
ENV_KEY = "SANDBOX_SNAPSHOT"


def pinned_requirements() -> list[str]:
    """["pandas==3.1.0", ...] with the versions locked in uv.lock."""
    with open(PROJECT_ROOT / "uv.lock", "rb") as f:
        locked = {package["name"]: package["version"] for package in tomllib.load(f)["package"]}
    return [f"{name}=={locked[name]}" for name in LIBRARIES]


def run(sandbox, command: str) -> str:
    """Run a command in the sandbox and return its output. Raises if it fails."""
    result = sandbox.run(command, timeout=STEP_TIMEOUT)
    if result.exit_code != 0:
        raise RuntimeError(f"`{command}` failed (exit code {result.exit_code}):\n{result.stderr or result.stdout}")
    return result.stdout


def save_snapshot_id(snapshot_id: str) -> None:
    """Set SANDBOX_SNAPSHOT=<id> in .env if it is missing or empty there. Only that line is touched."""
    new_line = f"{ENV_KEY}={snapshot_id}"
    lines = []
    if ENV_FILE.exists():
        with open(ENV_FILE, encoding="utf-8", newline="") as f:  # newline="" keeps the file's LF or CRLF
            lines = f.read().splitlines(keepends=True)
    for i, line in enumerate(lines):
        key, sep, value = line.partition("=")
        if not sep or key.strip() != ENV_KEY:
            continue
        if value.split("#", 1)[0].strip().strip("\"'"):  # already set, comments and empty quotes do not count
            print(f"{ENV_KEY} is already set in .env, left unchanged. To use the new snapshot change it to:  {new_line}")
            return
        lines[i] = new_line + line[len(line.rstrip("\r\n")) :]
        break
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line + "\n")
    ENV_FILE.write_text("".join(lines), encoding="utf-8", newline="")
    print(f"Wrote {new_line} to .env")


def main() -> None:
    name = f"snapshot-builder-{uuid.uuid4().hex[:8]}"
    try:
        print(f"Starting sandbox {name} ...")
        sandbox = client.create_sandbox(
            name=name, timeout=settings.sandbox_create_timeout, idle_ttl_seconds=SANDBOX_IDLE_TTL_SECONDS
        )

        # The default image only has `python3`, but the skills and the agent run `python`
        run(sandbox, 'ln -sf "$(command -v python3)" /usr/local/bin/python')

        requirements = pinned_requirements()
        print("Installing", " ".join(requirements), "...")
        # The image's Python is marked externally managed (PEP 668), so pip needs
        # --break-system-packages; fine in a throwaway image. --no-cache-dir keeps
        # pip's download cache out of the snapshot.
        run(sandbox, "python -m pip install -q --no-cache-dir --break-system-packages " + " ".join(requirements))
        print("Import check OK, versions:", run(sandbox, IMPORT_CHECK).strip())

        tag = date.today().isoformat()
        print(f"Capturing snapshot {SNAPSHOT_NAME}:{tag} ...")
        snapshot = client.capture_snapshot(name, SNAPSHOT_NAME, tag=tag, timeout=STEP_TIMEOUT)
        print(f"Snapshot id: {snapshot.id}")
        save_snapshot_id(snapshot.id)
    finally:
        # Also when create_sandbox failed: it may have timed out after the sandbox was created
        try:
            client.delete_sandbox(name)
            print(f"Deleted sandbox {name}")
        except ResourceNotFoundError:
            pass
        except Exception as error:  # never hide the error that brought us here
            print(f"Could not delete sandbox {name}, delete it by hand in LangSmith: {error}")


if __name__ == "__main__":
    main()
