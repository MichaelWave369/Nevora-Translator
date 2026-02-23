from __future__ import annotations

from translator.models import ParsedIntent


class GDScriptRenderer:
    name = "gdscript"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''# Prompt: {prompt}
# Mode: {mode}
extends Node

func _ready() -> void:
    print("Actions: {', '.join(intent.actions)}")
'''
