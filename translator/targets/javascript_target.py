from __future__ import annotations

from translator.models import ParsedIntent


class JavaScriptRenderer:
    name = "javascript"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''// Prompt: {prompt}\n// Mode: {mode}
class GeneratedFeature {{
  run() {{ console.log("Actions: {', '.join(intent.actions)}"); }}
}}
'''
