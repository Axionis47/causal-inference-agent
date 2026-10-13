"""What a lane writes back: the comparison, the checks, the estimates, the falsifications, the interpretation, the honest stop,
the declines, the context deficiency, and the record the desk keeps of a run."""

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
    design_id: str = ""
    contrast: str = Field(description="contrast key, or 'all' when the check does not depend on the contrast")
    name: str
    level: Literal["pass", "soft", "hard"]
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"check:{self.design_id + '.' if self.design_id else ''}{self.contrast}.{self.name}"


class Checks(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def flags(self) -> list[CheckResult]:
        return [r for r in self.results if r.level != "pass"]

    @property
    def hard(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "hard"]


class ConfidenceSet(BaseModel):
    status: Literal["bounded", "disconnected", "unresolved_tail", "empty", "computational_failure"]
    regions: list[list[float]] = Field(default_factory=list)
    searched_range: list[float] = Field(default_factory=list)
    resolution: float | None = None
    detail: str = ""


class Estimate(BaseModel):
    design_id: str = ""
    endpoint_id: str = ""
    target: dict = Field(default_factory=dict)
    confidence_set: ConfidenceSet | None = None
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
    modifier: str | None = Field(default=None, description="set when this is the effect within one level of a modifier column")
    level: str | None = Field(default=None, description="the modifier's level or bin this estimate is for")
    p_value: float | None = Field(default=None, description="the p-value the design's inference gives this estimate")
    p_value_source: str | None = Field(
        default=None,
        description="how the p-value was computed: the fit's own errors (CRV1, hetero, nn), a resample (ritest, a wild bootstrap), or a correction across estimates (rwolf)",
    )

    @property
    def tag(self) -> str:
        """The address stem: estimate:<contrast>, .<method> for a secondary, .by.<modifier>.<level> for a level of a modifier."""
        base = f"estimate:{self.design_id + '.' if self.design_id else ''}{self.contrast}" + (f".{self.method}" if self.secondary else "")
        return base + (f".by.{_slug(self.modifier)}.{_slug(self.level or '')}" if self.modifier is not None else "")


class Falsification(BaseModel):
    design_id: str = ""
    contrast: str
    refuter: str
    new_effect: float | None = None
    p_value: float | None = None
    passed: bool | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"falsification:{self.design_id}.{self.contrast}.{self.refuter}"


class Sensitivity(BaseModel):
    design_id: str = ""
    contrast: str
    method: str
    range_low: float | None = None
    range_high: float | None = None
    points: list[dict] = Field(default_factory=list)
    detail: str = ""
    error: str | None = None
    target: dict = Field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"sensitivity:{self.design_id}.{self.contrast}.{self.method}"


class Interpretation(BaseModel):
    design_id: str = ""
    contrast: str
    answer: str = Field(description="the answer to the question for this contrast, two or three sentences, in the outcome's units")
    effect_stated: float = Field(description="the effect size you are reporting, copied from the estimate")
    caveats: list[str] = Field(description="what the reader must know: the assumption bet on, flags, refuters that failed")
    cites: list[str] = Field(description="artifact addresses such as estimate:<contrast>.value, check:<contrast>.overlap, refute:<contrast>.<refuter>.p_value")


class Feasibility(BaseModel):
    """An honest stop. Which stage, why, the facts, and what would make the analysis possible."""

    terminal_kind: Literal["context_deficiency", "unidentified_effect", "unsupported_method", "inadequate_support", "exhausted_reasoning", "execution_failure"]
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
    figures: list[dict] = Field(default_factory=list, description="FigureSpecs the run left behind")
    what_if: dict[str, str] = Field(default_factory=dict, description="for a what-if design: the fields changed on the fork, address -> value")
    differs: list[str] = Field(default_factory=list, description="the fields that differ from the design before, by address")
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: str | None = None
    lead_design_id: str | None = None
    decision_record: str = ""
    decision: dict = Field(default_factory=dict)  # chosen, chosen_assumption, why, over: {family: reason}
    specialist_result: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)  # artifacts.json when the run dir holds one
