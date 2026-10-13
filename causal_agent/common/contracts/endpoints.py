"""Endpoint declarations and the evidence a context must carry before reasoning starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    address: str
    allowed_statuses: list[str] = Field(default_factory=lambda: ["confirmed"])
    allowed_sources: list[str] = Field(default_factory=lambda: ["user:", "doc:", "data", "code:", "fixture:", "eval:"])
    allow_unknown: bool = False
    allow_contradiction: bool = False
    when: dict[str, list[Any]] = Field(default_factory=dict)
    values: list[Any] = Field(default_factory=list)


class EndpointDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    targets: list[str]
    scope_targets: list[str] = Field(default_factory=lambda: ["average"])
    configuration: dict[str, Any] = Field(default_factory=dict)
    requirements: list[ContextRequirement] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    trajectory: list[str]
    checks: list[str] = Field(default_factory=list)
    estimators: list[str] = Field(default_factory=list)
    inference: list[str] = Field(default_factory=list)
    terminal_failures: list[str] = Field(default_factory=list)
    activation: dict[str, list[Any]] = Field(default_factory=dict)
    activation_required: bool = False
    executable: bool = True

    @model_validator(mode="after")
    def complete(self):
        if not self.trajectory or len(self.trajectory) != len(set(self.trajectory)):
            raise ValueError(f"{self.id}: a trajectory needs distinct decisions")
        if not self.targets:
            raise ValueError(f"{self.id}: no causal target declared")
        if self.executable and not self.estimators:
            raise ValueError(f"{self.id}: no execution capability declared")
        return self


class EndpointCatalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoints: list[EndpointDefinition]

    @model_validator(mode="after")
    def unique(self):
        ids = [e.id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def endpoint(self, endpoint_id: str) -> EndpointDefinition:
        return next(e for e in self.endpoints if e.id == endpoint_id)


def load_endpoints(path: str | Path) -> EndpointCatalogue:
    return EndpointCatalogue.model_validate(yaml.safe_load(Path(path).read_text()))


class Obligation(BaseModel):
    id: str
    address: str
    status: Literal["met", "assumed", "missing", "failed", "inapplicable"]
    reason: str
    cites: list[str] = Field(default_factory=list)


class EndpointAdmission(BaseModel):
    endpoint_id: str
    status: Literal["ready", "conditional", "missing", "excluded", "unsupported"]
    obligations: list[Obligation] = Field(default_factory=list)
    reason: str = ""

    @property
    def admitted(self) -> bool:
        return self.status in {"ready", "conditional"}


class ContextAdmission(BaseModel):
    family: str
    contract: str
    memory_version: int
    data_version: int
    endpoints: list[EndpointAdmission]

    @property
    def ready(self) -> bool:
        active = [e for e in self.endpoints if e.status not in {"excluded", "unsupported"}]
        return bool(active) and all(e.admitted for e in active)

    @property
    def missing(self) -> list[str]:
        return list(dict.fromkeys(o.address for e in self.endpoints if e.status == "missing" for o in e.obligations if o.status in {"missing", "failed"}))


class ContextDeficiency(BaseModel):
    stage: str
    addresses: list[str]
    decisions: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    reason: str


class CandidateRecord(BaseModel):
    id: str
    endpoint_id: str
    status: Literal["proposed", "ready", "excluded", "unsupported", "unidentified", "failed", "frozen"]
    reason: str = ""
    cites: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    obligations: list[Obligation] = Field(default_factory=list)


def assess_context(
    catalogue: EndpointCatalogue,
    facts: dict[str, Any],
    *,
    family: str,
    memory_version: int,
    data_version: int,
    target: str = "average",
    bindings: dict[str, str | list[str]] | None = None,
) -> ContextAdmission:
    """Check statuses and declared conditions. No model judgement or statistical truth claim."""
    bindings = bindings or {}

    def field(address: str):
        return facts.get(address)

    def value(address: str):
        f = field(address)
        return getattr(f, "value", None) if f is not None else None

    def matches(conditions: dict[str, list[Any]]) -> bool | None:
        undecided = False
        for address, allowed in conditions.items():
            v = value(address)
            if v is None:
                undecided = True
            elif allowed != ["*"] and str(v).lower() not in {str(a).lower() for a in allowed}:
                return False
        return None if undecided else True

    admissions = []
    for endpoint in catalogue.endpoints:
        if target not in endpoint.scope_targets and "*" not in endpoint.scope_targets:
            admissions.append(EndpointAdmission(endpoint_id=endpoint.id, status="excluded", reason=f"does not serve requested target {target}"))
            continue
        activation = matches(endpoint.activation)
        if activation is False or (activation is None and not endpoint.activation_required):
            admissions.append(EndpointAdmission(endpoint_id=endpoint.id, status="excluded", reason="configuration is not activated by supplied context"))
            continue
        obligations = []
        verdict: Literal["met", "assumed", "missing", "failed", "inapplicable"]
        expanded_requirements = []
        for requirement in endpoint.requirements:
            addresses = [requirement.address]
            for name, bound in bindings.items():
                choices = bound if isinstance(bound, list) else [bound]
                if any(f"<{name}>" in a for a in addresses):
                    if not choices:
                        obligations.append(
                            Obligation(
                                id=requirement.id,
                                address="claim:target.modifiers" if name == "modifier" else "claim:grain.row_is",
                                status="missing",
                                reason=f"the endpoint requires named {name} columns",
                            )
                        )
                        addresses = []
                    else:
                        addresses = [a.replace(f"<{name}>", choice) for a in addresses for choice in choices]
            expanded_requirements += [
                requirement.model_copy(update={"address": a, "id": requirement.id + (":" + a if a != requirement.address else "")}) for a in addresses
            ]
        for req in expanded_requirements:
            if matches(req.when) is False:
                continue
            address = req.address
            f = field(address)
            status = getattr(f, "status", "empty")
            source = getattr(f, "source", None) or ""
            from_model = source.startswith("model:")
            accepted_source = bool(source) and not from_model and any(source.startswith(prefix) for prefix in req.allowed_sources)
            if req.values and value(address) is not None and str(value(address)).lower() not in {str(v).lower() for v in req.values}:
                verdict, reason = "failed", f"supplied value {value(address)!r} excludes this configuration"
            elif status in req.allowed_statuses and value(address) is not None and accepted_source:
                verdict, reason = "met", f"{status} by {source or 'recorded source'}"
            elif status == "unknown" and req.allow_unknown and accepted_source:
                verdict, reason = "assumed", "the contract permits this explicit unknown"
            elif status == "contradiction" and req.allow_contradiction and accepted_source:
                verdict, reason = "assumed", "the contract permits retaining the conflicting accounts"
            elif status in {"refuted", "contradiction"}:
                verdict, reason = "failed", f"blocking {status}"
            else:
                verdict, reason = "missing", f"requires {req.allowed_statuses}; currently {status}"
            obligations.append(Obligation(id=req.id, address=address, status=verdict, reason=reason, cites=[address] if f is not None else []))
        state: Literal["ready", "conditional", "missing", "excluded", "unsupported"] = (
            "excluded"
            if any(o.reason.startswith("supplied value") for o in obligations)
            else "missing"
            if any(o.status in {"missing", "failed"} for o in obligations)
            else "conditional"
            if any(o.status == "assumed" for o in obligations)
            else "ready"
        )
        if activation is None and state in {"ready", "conditional"}:
            state = "missing"
            for address in endpoint.activation:
                if value(address) is None:
                    obligations.append(
                        Obligation(id="configuration", address=address, status="missing", reason="configuration requires this computed or supplied fact")
                    )
        admissions.append(EndpointAdmission(endpoint_id=endpoint.id, status=state, obligations=obligations))
    return ContextAdmission(family=family, contract=catalogue.fingerprint, memory_version=memory_version, data_version=data_version, endpoints=admissions)
