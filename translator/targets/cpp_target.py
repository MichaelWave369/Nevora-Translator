from __future__ import annotations

from translator.models import ParsedIntent


class CppRenderer:
    name = "cpp"

    def render(self, prompt: str, intent: ParsedIntent, mode: str = "gameplay", plan=None) -> str:
        return f'''// Prompt: {prompt}\n// Mode: {mode}
#include <iostream>
class GeneratedFeature {{
public:
  void Run() {{ std::cout << "Actions: {', '.join(intent.actions)}" << std::endl; }}
}};
'''
