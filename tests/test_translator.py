from translator.core import EnglishToCodeTranslator


def test_python_translation_contains_class_and_prompt() -> None:
    translator = EnglishToCodeTranslator()
    output = translator.translate(
        prompt="Create a player that can jump when space is pressed",
        target="python",
    )

    assert "class GeneratedFeature" in output
    assert "space_pressed" in output
    assert "Prompt:" in output


def test_blueprint_translation_contains_nodes() -> None:
    translator = EnglishToCodeTranslator()
    output = translator.translate(
        prompt="When health is zero, play death animation and disable input",
        target="blueprint",
    )

    assert "[Event BeginPlay]" in output
    assert "[Event Tick]" in output
    assert "Execute Actions" in output
