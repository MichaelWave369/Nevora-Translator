"""Prototype Unreal import script (run inside Unreal Editor Python environment).

Reads a graph contract JSON and prints the steps to create Blueprint nodes.
Replace print calls with unreal.EditorAssetLibrary / K2 graph APIs as needed.
"""

from __future__ import annotations

import json
from pathlib import Path


def import_graph_contract(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Importing Blueprint: {payload['blueprint_name']}")
    for node in payload.get("nodes", []):
        print(f"Create node: {node['type']} (id={node['id']})")
    for edge in payload.get("edges", []):
        print(f"Connect {edge['from']} -> {edge['to']}")


if __name__ == "__main__":
    import_graph_contract("artifacts/bp_graph.json")
