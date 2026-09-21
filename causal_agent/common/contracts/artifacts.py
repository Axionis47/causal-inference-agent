"""What a lane writes back: the comparison, the checks, the estimates, the falsifications, the interpretation, the honest stop,
the declines, the ask, and the record the desk keeps of a run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts.base import Cited, _slug

# ----------------------------------------------------------------- specialist artifacts
# Lane-invariant. Every specialist produces these shapes; lane-specific parts sit inside.


class Contrast(Cited):
    """One comparison: rows at `control` versus rows at `treated`."""

    control: str
    treated: str

    @property
    def key(self) -> str:
        return f"{_slug(self.treated)}_vs_{_slug(self.control)}"


class CheckResult(BaseModel):
    contrast: str = Field(description="contrast key, or 'all' when the check does not depend on the contrast")
    name: str
    level: Literal["pass", "soft", "hard"]
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"check:{self.contrast}.{self.name}"


class Checks(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def flags(self) -> list[CheckResult]:
        return [r for r in self.results if r.level != "pass"]

    @property
    def hard(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "hard"]


class Estimate(BaseModel):
    contrast: str
    method: str
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_treated: int = 0
    n_control: int = 0
    target_units: str = "ate"
    error: str | None = None
    secondary: bool = False


class Refutation(BaseModel):
    contrast: str
    refuter: str
    kind: Literal["falsification", "sensitivity"]
    new_effect: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    p_value: float | None = None
    passed: bool | None = Field(default=None, description="None for sensitivity; it reports a range, not a verdict")
    detail: str = ""


class Interpretation(BaseModel):
    contrast: str
    answer: str = Field(description="the answer to the question for this contrast, two or three sentences, in the outcome's units")
    effect_stated: float = Field(description="the effect size you are reporting, copied from the estimate")
    caveats: list[str] = Field(description="what the reader must know: the assumption bet on, flags, refuters that failed")
    cites: list[str] = Field(description="artifact addresses such as estimate:<contrast>.value, check:<contrast>.overlap, refute:<contrast>.<refuter>.p_value")


class Feasibility(BaseModel):
    """An honest stop. Which stage, why, the facts, and what would make the analysis possible."""

    stage: str
    reason: str
    facts: list[str] = Field(default_factory=list)
    what_would_fix: str = ""


class Decline(BaseModel):
    """The lane did not take a pack field as given. Every override of the pack is one of these, with an address, so the
    brief and the page can show where the lane and the desk disagreed."""

    stage: str = Field(description="the node that declined")
    kind: Literal["declined", "replaced", "substituted"] = Field(
        description="declined: could not apply it; replaced: took another value; substituted: answered a near thing"
    )
    about: str = Field(description="the pack address: scope.window, design.cluster_level, claim:assignment.cutoff, col:<key>")
    pack_value: str | None = None
    took: str | None = None
    reason: str
    check: str = Field(description="the code rule that decided: intake.filter_unparsed, groups.level_observed, shape.pre_periods")
    cites: list[str] = Field(default_factory=list)

    @property
    def address(self) -> str:
        return f"decline:{self.stage}.{_slug(self.about)}"

    def render(self) -> str:
        pack = f" · pack said {self.pack_value!r}" if self.pack_value is not None else ""
        took = f" · lane took {self.took!r}" if self.took is not None else ""
        return f"[{self.address}] {self.kind}: {self.about}{pack}{took} · {self.reason} ({self.check})"


class LaneAsk(BaseModel):
    """One question back to the desk, keyed to a memory address the desk can settle. The desk asks it as it asks
    everything else, and the lane runs again on the memory as it then stands."""

    address: str
    question: str
    options: list[str] = Field(default_factory=list)
    because: str = Field(default="", description="the check or belief that opened the question, in words")
    evidence: list[str] = Field(default_factory=list, description="check addresses shown beside the question")
    stage: str = ""


class RunRecord(BaseModel):
    """What one run left behind, as the desk keeps it in state and the server shows it."""

    index: int
    dataset: str
    question: str
    family: str | None = None
    specialist: str | None = None
    status: str = "no_handoff"
    run_dir: str | None = None
    design_dir: str | None = None
    figures: list[dict] = Field(default_factory=list, description="FigureSpecs the run left behind, the ready-moment figure first")
    what_if: dict[str, str] = Field(default_factory=dict, description="for a what-if design: the fields changed on the fork, address -> value")
    differs: list[str] = Field(default_factory=list, description="the fields that differ from the design before, by address")
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: str | None = None
    decision_record: str = ""
    decision: dict = Field(default_factory=dict)  # chosen, chosen_assumption, why, over: {family: reason}
    specialist_result: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)  # artifacts.json when the run dir holds one
