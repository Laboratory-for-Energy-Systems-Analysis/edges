# conftest.py
import os
from pathlib import Path
import re
import json
import sys
import platform
import pytest

BASE = Path(os.environ.get("PYTEST_DEBUG_DIR", ".pytest-debug"))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


@pytest.fixture(scope="session", autouse=True)
def debug_base():
    BASE.mkdir(parents=True, exist_ok=True)
    # Snapshot environment info once per session
    (BASE / "env.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "env": dict(os.environ),
            },
            indent=2,
        )
    )
    return BASE


@pytest.fixture
def test_debug_dir(request, debug_base: Path):
    d = debug_base / _slug(request.node.nodeid)
    d.mkdir(parents=True, exist_ok=True)
    return d
