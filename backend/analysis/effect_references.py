"""Carry accepted effect identity forward without re-authoring its source claim."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from backend.analysis.contracts import Target


def decoder(target: dict[str, Any], prerequisites: list[str]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Capture the offered accepted target before decoding provider-local aliases."""
    accepted = Target.model_validate(target)
    registry = {effect.id: effect for effect in accepted.requested_effects}
    if (
        "decision:target" not in prerequisites
        or accepted.status != "ready"
        or not registry
        or len(registry) != len(accepted.requested_effects)
        or sum(effect.priority == "primary" for effect in registry.values()) != 1
        or any(effect.question.basis != "supplied" or not effect.question.references for effect in registry.values())
    ):
        raise ValueError("Effect references require one accepted target with unique supplied questions and exactly one primary effect.")
    fields = {"id", "treatment_ids", "estimand", "comparison"}

    def decode(proposal: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(proposal)
        effects = result.get("effects", [])
        if not isinstance(effects, list):
            raise ValueError("Effect reference output must contain an effects list.")
        seen: set[str] = set()
        for effect in effects:
            if not isinstance(effect, dict) or set(effect) != fields:
                raise ValueError("Each effect must contain exactly id, treatment_ids, estimand and comparison. Do not re-author the accepted contrast or priority.")
            identity = effect["id"]
            if not isinstance(identity, str) or identity not in registry or identity in seen:
                raise ValueError("Effect references must select distinct exact accepted target identities.")
            seen.add(identity)
            selected = registry[identity]
            claim = selected.question.model_dump(mode="json")
            claim["prerequisites"] = list(dict.fromkeys([*claim["prerequisites"], "decision:target"]))
            effect.update(contrast=claim, priority=selected.priority)
        if result.get("status") == "ready" and seen != registry.keys():
            raise ValueError("Ready treatment planning must bind every accepted requested effect exactly once.")
        return result

    return decode
