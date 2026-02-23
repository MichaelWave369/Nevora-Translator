from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ParsedIntent:
    entities: List[str]
    actions: List[str]
    conditions: List[str]
    outputs: List[str] = field(default_factory=list)


@dataclass
class IntentSchema:
    """Canonical planner schema used by every planner path."""

    entities: List[str]
    actions: List[str]
    conditions: List[str]
    outputs: List[str]


@dataclass
class PlanStep:
    name: str
    details: str


@dataclass
class EventSpec:
    name: str
    trigger: str


@dataclass
class StateTransition:
    from_state: str
    to_state: str
    condition: str


@dataclass
class GenerationIR:
    events: List[EventSpec]
    transitions: List[StateTransition]
    side_effects: List[str]
    error_branches: List[str]


@dataclass
class GenerationPlan:
    intent: ParsedIntent
    ir: GenerationIR
    steps: List[PlanStep]
    state_model: Dict[str, str]
