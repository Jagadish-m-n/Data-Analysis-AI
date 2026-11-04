"""Tests for app/sandbox/sandboxbackend.py. They run offline: no real sandbox is created.

Run with:  uv run pytest tests/test_sandbox.py -v
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.sandbox import sandboxbackend as sandbox


class FakeBackend:
    """Stands in for LangSmithSandbox: returns canned results instead of calling LangSmith."""

    id = "fake-sandbox"

    def __init__(self, entries=(), downloads=(), output="", exit_code=0):
        self.entries = list(entries)
        self.downloads = list(downloads)
        self.output = output
        self.exit_code = exit_code
        self.commands = []

    async def als(self, path):
        return SimpleNamespace(error=None, entries=self.entries)

    async def adownload_files(self, paths):
        return [download for download in self.downloads if download.path in paths]

    async def aexecute(self, command):
        self.commands.append(command)
        return SimpleNamespace(output=self.output, exit_code=self.exit_code)


@pytest.fixture
def uploaded(monkeypatch):
    """Records what would be uploaded to the sandbox instead of uploading it."""
    files = []

    async def fake_upload(backend, new_files):
        files.extend(new_files)

    monkeypatch.setattr(sandbox, "_upload", fake_upload)
    return files


@pytest.mark.parametrize(
    "name, expected",
    [
        ("chart.png", "image/png"),
        ("chart.SVG", "image/svg+xml"),
        ("report.md", "text/markdown"),
        ("clean.csv", "text/csv"),
        ("model.pkl", None),
        ("README", None),
    ],
)
def test_output_type(name, expected):
    assert sandbox.output_type(name) == expected


def test_upload_data_files_uses_data_folder(uploaded):
    paths = asyncio.run(sandbox.upload_data_files(FakeBackend(), [("sales.csv", b"a,b\n1,2\n")]))

    assert paths == ["/data/sales.csv"]
    assert uploaded == [("/data/sales.csv", b"a,b\n1,2\n")]


def test_upload_data_files_creates_output_folder(uploaded):
    backend = FakeBackend()
    asyncio.run(sandbox.upload_data_files(backend, [("sales.csv", b"a,b\n1,2\n")]))

    assert backend.commands == ["mkdir -p /output/"]


def test_upload_data_files_fails_if_output_folder_cannot_be_created(uploaded):
    backend = FakeBackend(output="mkdir: Permission denied", exit_code=1)

    with pytest.raises(RuntimeError, match="/output/"):
        asyncio.run(sandbox.upload_data_files(backend, [("sales.csv", b"a,b\n1,2\n")]))


def test_upload_skills_uses_skills_folder(uploaded):
    asyncio.run(sandbox.upload_skills(FakeBackend()))

    paths = [path for path, _ in uploaded]
    assert "/skills/data-analysis/SKILL.md" in paths
    assert all(path.startswith("/skills/") for path in paths)


def test_download_output_files_returns_supported_files():
    backend = FakeBackend(
        entries=[
            {"path": "/output/chart.png"},
            {"path": "/output/model.pkl"},
            {"path": "/output/report.md"},
            {"path": "/output/tmp", "is_dir": True},
        ],
        downloads=[
            SimpleNamespace(path="/output/chart.png", content=b"png-bytes"),
            SimpleNamespace(path="/output/report.md", content=None),  # download failed
        ],
    )

    files, warnings = asyncio.run(sandbox.download_output_files(backend))

    assert files == [("chart.png", b"png-bytes")]
    assert len(warnings) == 2
    assert "model.pkl" in warnings[0]
    assert "report.md" in warnings[1]


def test_download_output_files_with_no_output():
    assert asyncio.run(sandbox.download_output_files(FakeBackend())) == ([], [])


def test_list_tree_parses_find_output():
    backend = FakeBackend(output="/data/sales.csv\tf\t120\n/output/charts\td\t4096\nnot a valid line\n")

    assert asyncio.run(sandbox.list_tree(backend)) == [
        {"path": "/data/sales.csv", "type": "file", "size": 120},
        {"path": "/output/charts", "type": "dir", "size": 4096},
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
