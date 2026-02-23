from __future__ import annotations

import argparse

from .core import EnglishToCodeTranslator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English-to-code translator")
    parser.add_argument("--target", required=True, choices=["python", "blueprint"])
    parser.add_argument("--prompt", required=True, help="English description to translate")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    translator = EnglishToCodeTranslator()
    print(translator.translate(prompt=args.prompt, target=args.target))


if __name__ == "__main__":
    main()
