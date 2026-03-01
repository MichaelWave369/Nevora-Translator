from __future__ import annotations

from pathlib import Path
import zipfile

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older environments
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

INCLUDE_PATHS = [
    "translator",
    "app.py",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "requirements-ui.txt",
    "requirements-providers.txt",
]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".venv", "venv"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _load_project_version() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)

    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("Unable to read [project].version from pyproject.toml")
    return version.strip()


def _should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def main() -> None:
    DIST.mkdir(exist_ok=True)
    version = _load_project_version()
    out = DIST / f"nevora-translator-{version}.zip"

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            source = ROOT / rel
            if source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file() and _should_include(file.relative_to(ROOT)):
                        zf.write(file, file.relative_to(ROOT))
            elif source.exists() and _should_include(source.relative_to(ROOT)):
                zf.write(source, source.relative_to(ROOT))

    print(f"Release package created: {out}")


if __name__ == "__main__":
    main()
