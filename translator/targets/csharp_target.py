from __future__ import annotations

from translator.models import ParsedIntent


class CSharpRenderer:
    name = "csharp"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''// Prompt: {prompt}\n// Mode: {mode}
using System;
public class GeneratedFeature {{
  public void Run() {{ Console.WriteLine("Actions: {', '.join(intent.actions)}"); }}
}}
'''
