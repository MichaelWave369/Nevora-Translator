from __future__ import annotations

from translator.models import ParsedIntent


class BlueprintRenderer:
    name = "blueprint"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''# Unreal Engine Blueprint-style pseudograph (beginner-friendly)
# Prompt: {prompt}
# Mode: {mode}
# Read top-to-bottom as execution flow.

[Event BeginPlay]
  -> [Comment: Initialize core entities extracted from prompt]
  -> [Set Entities = "{', '.join(intent.entities)}"]
  -> [Comment: Check conditions before doing actions]
  -> [Branch: {', '.join(intent.conditions)}]
      True -> [Execute Actions: {', '.join(intent.actions)}]
      False -> [No-op]
'''
