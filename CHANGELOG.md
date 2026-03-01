# Changelog

## 0.1.0rc1 - 2026-03-01

### Added
- Proper Python packaging with `pyproject.toml` and console script (`nevora-translator`).
- Canonical package version metadata in `translator/_version.py`.
- Dependency split by use-case (`core`, `ui`, `providers`, `planners`, `dev`).
- GitHub Actions CI workflow for tests across Python 3.10/3.11.
- GitHub Actions tag-based release workflow with zip artifact attachment.
- Release packaging script at `scripts/package_release.py`.
- `.env.example` for provider and export configuration.
- Smoke tests for CLI and Streamlit startup import path.

### Changed
- Removed `sys.path` import hack from eval runner and made `eval` importable package.
- Updated README with release-ready install/run/test/fallback documentation.

### Notes
- This is a release candidate intended for first public distribution.
