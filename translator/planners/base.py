from __future__ import annotations

from typing import Protocol

from translator.models import ParsedIntent


class SemanticPlanner(Protocol):
    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        ...
