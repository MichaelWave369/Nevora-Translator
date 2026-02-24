from __future__ import annotations

from translator.models import GenerationPlan, ParsedIntent


class PythonRenderer:
    name = "python"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan: GenerationPlan | None = None) -> str:
        actions = ", ".join(intent.actions)
        return f'''"""Beginner-friendly generated starter.

What this file contains:
1) A short metadata block so you know where this came from.
2) A `GeneratedFeature` class you can rename.
3) A `run(...)` method where your event logic executes.

Prompt: {prompt}
Mode: {mode}
IR: {plan.ir.side_effects if plan else []}
"""

# Standard library import used for type hints in this starter.
from typing import Dict, Any


class GeneratedFeature:
    """A simple feature container.

    - `entities` tracks the core game/app objects this prompt mentioned.
    - `outputs` tracks expected output channels (state/log/ui/etc).
    """

    def __init__(self):
        # Store parsed intent so beginners can inspect what was extracted.
        self.entities = {intent.entities!r}
        self.outputs = {intent.outputs!r}

    def run(self, event: Dict[str, Any]) -> None:
        """Handle one incoming event.

        Expected event example:
        {{"type": "input", "key": "Space"}}
        """
        # Guard clause: only process known event categories.
        if event.get("type") not in ("input", "tick", "request"):
            return

        # Main action line generated from your prompt.
        print("Actions: {actions}")


if __name__ == "__main__":
    # Quick beginner demo entrypoint.
    feature = GeneratedFeature()
    feature.run({{"type": "input", "key": "Space"}})
'''
