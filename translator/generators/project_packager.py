from __future__ import annotations

import io
import zipfile
from typing import Dict


TARGET_FILENAMES = {
    "python": "main.py",
    "javascript": "main.js",
    "typescript": "main.ts",
    "cpp": "main.cpp",
    "csharp": "Program.cs",
    "gdscript": "main.gd",
    "blueprint": "main.txt",
}


def beginner_readme_text(project_title: str, app_summary: str, code_file: str) -> str:
    return (
        f"# {project_title}\n\n"
        "Hi! 👋 This project was generated for you by Nevora Translator.\n\n"
        "## What your app does\n"
        f"{app_summary}\n\n"
        "## How to run it in 3 simple steps\n"
        "1. Install Python from https://www.python.org/downloads/ (if you do not already have it).\n"
        "2. Open a terminal in this folder and run: `pip install -r requirements.txt`.\n"
        f"3. Start the app by running: `python {code_file}` (or double-click `run.bat` on Windows / `run.sh` on Mac/Linux).\n\n"
        "If something does not work, do not worry—read the error message and try again. "
        "Most issues are fixed by installing requirements and making sure you are in this folder.\n"
    )


def build_requirements_text(include_pygame: bool = False) -> str:
    packages = ["python-dotenv>=1.0.0"]
    if include_pygame:
        packages.append("pygame>=2.6.0")
    return "\n".join(packages) + "\n"


def build_run_scripts(code_file: str) -> Dict[str, str]:
    run_bat = (
        "@echo off\n"
        "echo Starting your app...\n"
        "python -m pip install -r requirements.txt\n"
        f"python {code_file}\n"
        "pause\n"
    )
    run_sh = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "echo 'Starting your app...'\n"
        "python3 -m pip install -r requirements.txt\n"
        f"python3 {code_file}\n"
    )
    return {"run.bat": run_bat, "run.sh": run_sh}


def package_single_file_project(
    code: str,
    prompt: str,
    target: str = "python",
    project_title: str = "Nevora Generated Project",
    include_pygame: bool = False,
) -> bytes:
    code_file = TARGET_FILENAMES.get(target.lower(), "main.txt")
    readme = beginner_readme_text(project_title=project_title, app_summary=prompt, code_file=code_file)
    scripts = build_run_scripts(code_file)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(code_file, code)
        zf.writestr("README.txt", readme)
        zf.writestr("requirements.txt", build_requirements_text(include_pygame=include_pygame))
        zf.writestr("run.bat", scripts["run.bat"])
        zf.writestr("run.sh", scripts["run.sh"])
    return buf.getvalue()


def package_world_builder_project(
    files: Dict[str, str],
    project_title: str,
    app_summary: str,
) -> bytes:
    readme = beginner_readme_text(project_title=project_title, app_summary=app_summary, code_file="main.py")
    scripts = build_run_scripts("main.py")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
        zf.writestr("README.txt", readme)
        zf.writestr("requirements.txt", build_requirements_text(include_pygame=True))
        zf.writestr("run.bat", scripts["run.bat"])
        zf.writestr("run.sh", scripts["run.sh"])
    return buf.getvalue()
