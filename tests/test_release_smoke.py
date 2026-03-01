from __future__ import annotations

import importlib.util
import subprocess
import sys


def test_cli_basic_translation_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "translator.cli",
            "--target",
            "python",
            "--prompt",
            "Create a player that can jump",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "GeneratedFeature" in proc.stdout


def test_streamlit_app_startup_import_smoke() -> None:
    if importlib.util.find_spec("streamlit") is None:
        return
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", "app.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
