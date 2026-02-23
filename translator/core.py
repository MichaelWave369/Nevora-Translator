from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ParsedIntent:
    entities: List[str]
    actions: List[str]
    conditions: List[str]


class EnglishToCodeTranslator:
    """Translate English product/game ideas into target code formats.

    This MVP intentionally uses deterministic parsing + templates so outputs are
    reproducible and easy to extend.
    """

    SUPPORTED_TARGETS = {"python", "blueprint"}

    def translate(self, prompt: str, target: str) -> str:
        normalized_target = target.strip().lower()
        if normalized_target not in self.SUPPORTED_TARGETS:
            supported = ", ".join(sorted(self.SUPPORTED_TARGETS))
            raise ValueError(f"Unsupported target '{target}'. Supported: {supported}")

        intent = self._parse_prompt(prompt)

        if normalized_target == "python":
            return self._render_python(prompt, intent)

        return self._render_blueprint(prompt, intent)

    def _parse_prompt(self, prompt: str) -> ParsedIntent:
        lower = prompt.lower()

        entities = [
            word
            for word in ["player", "enemy", "camera", "health", "score", "video"]
            if word in lower
        ]

        actions = [
            word
            for word in [
                "jump",
                "move",
                "shoot",
                "spawn",
                "play",
                "record",
                "save",
                "disable",
            ]
            if word in lower
        ]

        conditions = [
            phrase
            for phrase in [
                "when",
                "if",
                "on key",
                "pressed",
                "zero",
                "collision",
            ]
            if phrase in lower
        ]

        return ParsedIntent(
            entities=entities or ["system"],
            actions=actions or ["process"],
            conditions=conditions or ["always"],
        )

    def _render_python(self, prompt: str, intent: ParsedIntent) -> str:
        class_name = "GeneratedFeature"
        action_comment = ", ".join(intent.actions)
        entity_comment = ", ".join(intent.entities)
        condition_comment = ", ".join(intent.conditions)

        return f'''"""Auto-generated from English prompt.
Prompt: {prompt}
"""

class {class_name}:
    def __init__(self):
        self.entities = {intent.entities!r}
        self.state = {{"active": True}}

    def run(self, event: dict) -> None:
        """Main behavior loop."""
        # Entities detected: {entity_comment}
        # Actions detected: {action_comment}
        # Conditions detected: {condition_comment}

        if event.get("type") in ("input", "tick", "collision"):
            self.handle_event(event)

    def handle_event(self, event: dict) -> None:
        if event.get("name") == "space_pressed" and "jump" in {intent.actions!r}:
            print("Player jump triggered")

        if event.get("health") == 0 and "disable" in {intent.actions!r}:
            self.state["active"] = False
            print("Feature deactivated")


if __name__ == "__main__":
    feature = {class_name}()
    feature.run({{"type": "input", "name": "space_pressed"}})
'''

    def _render_blueprint(self, prompt: str, intent: ParsedIntent) -> str:
        entities = ", ".join(intent.entities)
        actions = ", ".join(intent.actions)
        conditions = ", ".join(intent.conditions)

        return f'''# Unreal Engine Blueprint-style pseudograph
# Prompt: {prompt}

[Event BeginPlay]
    -> [Set Entities = "{entities}"]
    -> [Bind Input + Gameplay Events]

[Event InputAction Triggered]
    -> [Branch: Conditions include "{conditions}"]
        True -> [Execute Actions: {actions}]
        False -> [No Operation]

[Event Tick]
    -> [Update Runtime State]
    -> [Check Gameplay Conditions]
        -> [Branch]
            True -> [Apply Actions: {actions}]
            False -> [Continue]
'''
