from pathlib import Path

from eval.run_eval import _dataset_path, _simple_bleu


def test_dataset_path_exists() -> None:
    path = _dataset_path()
    assert isinstance(path, Path)
    assert path.exists()


def test_simple_bleu_nonzero_for_overlap() -> None:
    score = _simple_bleu("player jump", "player jump now")
    assert score > 0
