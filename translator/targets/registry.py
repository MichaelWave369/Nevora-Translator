from __future__ import annotations

from translator.targets.blueprint_target import BlueprintRenderer
from translator.targets.cpp_target import CppRenderer
from translator.targets.csharp_target import CSharpRenderer
from translator.targets.gdscript_target import GDScriptRenderer
from translator.targets.javascript_target import JavaScriptRenderer
from translator.targets.python_target import PythonRenderer


def build_registry() -> dict[str, object]:
    renderers = [
        PythonRenderer(),
        BlueprintRenderer(),
        CppRenderer(),
        CSharpRenderer(),
        JavaScriptRenderer(),
        GDScriptRenderer(),
    ]
    return {renderer.name: renderer for renderer in renderers}
