from __future__ import annotations

from translator.models import ParsedIntent


class BlueprintRenderer:
    name = "blueprint"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''# Unreal Engine Blueprint-style pseudograph
# Prompt: {prompt}
# Mode: {mode}

[Event BeginPlay]
  -> [Set Entities = "{', '.join(intent.entities)}"]
  -> [Branch: {', '.join(intent.conditions)}]
      True -> [Execute Actions: {', '.join(intent.actions)}]
'''
