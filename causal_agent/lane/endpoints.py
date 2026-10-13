"""Shared endpoint admission, identities, and trajectory verification."""

from __future__ import annotations

import hashlib
import json

from causal_agent.common.addresses import key
from causal_agent.common.contracts import ContextDeficiency, EndpointCatalogue, Handoff, assess_context
from causal_agent.profile.datasets import csv_path

MAX_DESIGNS = 8


def dataset_identity(h: Handoff) -> str:
    path = csv_path(h.pack_name, h.csv)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else ""


def context_identity(h: Handoff) -> str:
    data = h.model_dump(mode="json", exclude={"context_id", "design_id"})
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def admit(h: Handoff, catalogue: EndpointCatalogue) -> Handoff:
    bindings: dict[str, str | list[str]] = {
        "outcome": key(h.outcome),
        "treatment": key(h.treatment or ""),
        "modifier": [key(c) for c in h.scope.modifiers],
        "column": [c.key for c in h.columns if c.role not in {"outcome", "treatment", "score"}],
    }
    for name, field in (("score", "score_column"), ("mediator", "mediator_column"), ("instrument", "instrument_column")):
        value = h.assignment.get(field)
        if value:
            bindings[name] = key(str(value))
    for kind in ("mediator", "exclusion"):
        b = h.beliefs.get(kind)
        if b and b.column:
            bindings["mediator" if kind == "mediator" else "instrument"] = key(b.column)
    admission = assess_context(
        catalogue, h.facts, family=h.family, memory_version=h.memory_version, data_version=h.data_version, target=h.scope.target, bindings=bindings
    )
    pack = h.model_copy(update={"admission": admission, "dataset_identity": dataset_identity(h)})
    return pack.model_copy(update={"context_id": context_identity(pack)})


def validate_context(h: Handoff, catalogue: EndpointCatalogue) -> ContextDeficiency | None:
    if not h.dataset_identity or h.dataset_identity != dataset_identity(h):
        return ContextDeficiency(stage="load", addresses=["dataset.identity"], reason="the dataset bytes differ from the authorized context snapshot")
    if h.admission is None:
        return ContextDeficiency(stage="load", addresses=["context.admission"], reason="the desk has not admitted this context")
    if h.admission.contract != catalogue.fingerprint or h.context_id != context_identity(h):
        return ContextDeficiency(stage="load", addresses=["context.identity"], reason="context or endpoint requirements differ from the admitted snapshot")
    expected = admit(h, catalogue).admission
    if expected != h.admission:
        return ContextDeficiency(
            stage="load",
            addresses=expected.missing if expected else ["context.admission"],
            reason="recorded admission does not match the source context requirements",
        )
    if not h.admission.ready:
        return ContextDeficiency(stage="load", addresses=h.admission.missing, reason="required endpoint context is unresolved")
    return None


def design_identity(endpoint_id: str, configuration: dict, context_id: str) -> str:
    raw = json.dumps({"endpoint": endpoint_id, "configuration": configuration, "context": context_id}, sort_keys=True, default=str)
    return "d" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def verify_catalogue(catalogue: EndpointCatalogue, spec, estimator_names: set[str]) -> list[str]:
    errors: list[str] = []
    for endpoint in catalogue.endpoints:
        errors.extend(f"{endpoint.id}: unknown trajectory rung {r}" for r in endpoint.trajectory if r not in spec.names)
        errors.extend(f"{endpoint.id}: unimplemented estimator {e}" for e in endpoint.estimators if e not in estimator_names and e != "structural_zero")
        order = [spec.names.index(r) for r in endpoint.trajectory if r in spec.names]
        if order != sorted(order):
            errors.append(f"{endpoint.id}: trajectory does not follow the declared ladder")
        if endpoint.executable and spec.goal not in endpoint.trajectory:
            errors.append(f"{endpoint.id}: trajectory does not reach {spec.goal}")
    return errors
