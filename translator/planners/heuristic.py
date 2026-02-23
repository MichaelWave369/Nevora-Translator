from __future__ import annotations

from translator.models import ParsedIntent

MODE_LEXICONS = {
    "gameplay": {
        "entities": ["player", "enemy", "camera", "health", "weapon", "npc", "world"],
        "actions": ["jump", "move", "shoot", "spawn", "attack", "collect", "disable", "play"],
        "conditions": ["when", "if", "collision", "overlap", "timer", "pressed", "zero"],
        "outputs": ["animation", "sound", "ui", "state", "log"],
    },
    "automation": {
        "entities": ["file", "folder", "report", "email", "task", "system"],
        "actions": ["save", "send", "read", "parse", "transform", "schedule", "validate"],
        "conditions": ["when", "if", "daily", "hourly", "on failure", "on success"],
        "outputs": ["file", "log", "state", "notification"],
    },
    "video-processing": {
        "entities": ["video", "frame", "audio", "subtitle", "timeline"],
        "actions": ["render", "trim", "encode", "record", "overlay", "export"],
        "conditions": ["when", "if", "per frame", "on end", "start", "stop"],
        "outputs": ["file", "log", "preview"],
    },
    "web-backend": {
        "entities": ["request", "response", "user", "database", "session", "api"],
        "actions": ["validate", "save", "load", "authenticate", "authorize", "respond"],
        "conditions": ["when", "if", "on request", "on error", "unauthorized"],
        "outputs": ["json", "log", "state"],
    },
}


class HeuristicPlanner:
    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        lower = prompt.lower()
        lex = MODE_LEXICONS.get(mode, MODE_LEXICONS["gameplay"])

        def pick(bucket: str, fallback: str) -> list[str]:
            result = [token for token in lex[bucket] if token in lower]
            return result or [fallback]

        return ParsedIntent(
            entities=pick("entities", "system"),
            actions=pick("actions", "process"),
            conditions=pick("conditions", "always"),
            outputs=pick("outputs", "state"),
        )
