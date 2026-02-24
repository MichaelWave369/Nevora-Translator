# Example: English prompt to Python output

## Prompt
"When player presses space, make player jump and play a sound"

## Target
`python`

## Generated Output (truncated)
```python
class GeneratedFeature:
    def __init__(self):
        self.entities = ['player']

    def run(self, event: dict) -> None:
        if event.get("type") in ("input", "tick", "request"):
            print("Actions: jump, play")
```
