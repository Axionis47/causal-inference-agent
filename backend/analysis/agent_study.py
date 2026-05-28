"""Study agent tools assemble small source-bound records into one validated handoff."""

from typing import Any

from pydantic import Field

from backend.analysis import authority, contrasts, domain
from backend.analysis.agent_context import Publish, Workspace, decision
from backend.analysis.agent_review import require_review
from backend.analysis.agent_runtime import Tool
from backend.analysis.contracts import (
    AssignmentProcess,
    Claim,
    ContrastBinding,
    EffectRequest,
    RequestedEffect,
    Target,
    Treatment,
    Treatments,
    TreatmentValue,
    Variable,
)
from backend.validation import Contract


class TargetInput(Contract):
    outcome: Variable
    measurement: Claim
    population: Claim
    unit: Claim
    timing: Claim
    requested_effects: list[RequestedEffect] = Field(min_length=1, max_length=20)
    summary: str = Field(min_length=1, max_length=2000)


class FactorInput(Contract):
    treatment: Treatment
    values: list[TreatmentValue] = Field(default_factory=list, max_length=100)


class AssignmentInput(Contract):
    process: AssignmentProcess


class ComparisonInput(Contract):
    effect_id: str
    treatment_ids: list[str] = Field(min_length=1, max_length=20)
    estimand: str = Field(min_length=1, max_length=3000)
    comparison: ContrastBinding


class Study(Workspace):
    def target(self) -> Target:
        record = self.records().get("target")
        if record is None:
            raise ValueError("Set and validate the target before treatment or comparison records.")
        return Target.model_validate(record)

    def admit(self, value: Treatments) -> Treatments:
        prior = {"target": self.target().model_dump(mode="json")}
        self.check_exposure(value.model_dump(mode="json"))
        admitted, notes = authority.admit(value, self.context, prior, self.run.question, "treatments")
        try:
            authority.validate(admitted, prior, phase="treatments")
        except ValueError as error:
            raise ValueError(str(error) + "\n" + "\n".join(notes)) from error
        return admitted

    def set_target(self, args: TargetInput, key: str) -> dict[str, Any]:
        brief = self.context.analysis_brief
        if brief is None or brief.target is None or set(args.outcome.column_ids) != set(brief.target.column_ids):
            raise ValueError("The outcome must preserve the exact variable grouping in the applied brief.")
        fields = args.model_dump(mode="json", exclude={"summary"})
        candidate = Target.model_validate(decision(args.summary, effect_question=self.run.question, **fields))
        self.check_exposure(candidate.model_dump(mode="json"))
        accepted, notes = authority.admit(candidate, self.context, {}, self.run.question, "target")
        try:
            self.validate(accepted, {})
        except ValueError as error:
            raise ValueError(str(error) + "\n" + "\n".join(notes)) from error
        return self.save("target", accepted.model_dump(mode="json"), invalidate=("factor:", "assignment:", "effect:"))

    def set_factor(self, args: FactorInput, key: str) -> dict[str, Any]:
        brief = self.context.analysis_brief
        columns = sorted(args.treatment.variable.column_ids)
        if brief is None or columns not in [sorted(item.column_ids) for item in brief.treatments]:
            raise ValueError("A factor must preserve one exact treatment-variable grouping in the applied brief.")
        domain.validate_variable(args.treatment.variable, self.context)
        candidate = Treatments.model_validate(decision("Bound intervention factor and source-defined values.",
            treatments=[args.treatment.model_dump(mode="json")], values=[item.model_dump(mode="json") for item in args.values]))
        accepted = self.admit(candidate)
        return self.save("factor:" + args.treatment.variable.id,
                         {"treatment": accepted.treatments[0].model_dump(mode="json"), "values": [item.model_dump(mode="json") for item in accepted.values]},
                         invalidate=("assignment:", "effect:"))

    def set_assignment(self, args: AssignmentInput, key: str) -> dict[str, Any]:
        factors = {record["treatment"]["variable"]["id"] for name, record in self.records().items() if name.startswith("factor:")}
        if not set(args.process.treatment_ids) <= factors:
            raise ValueError("Set the referenced factors before their assignment process.")
        accepted = self.admit(Treatments.model_validate(decision("Bound assignment process.", assignments=[args.process.model_dump(mode="json")])))
        return self.save("assignment:" + args.process.id, accepted.assignments[0].model_dump(mode="json"))

    def set_comparison(self, args: ComparisonInput, key: str) -> dict[str, Any]:
        target = self.target()
        requested = next((item for item in target.requested_effects if item.id == args.effect_id), None)
        if requested is None:
            raise ValueError("Select an existing requested effect identity. Auxiliary calculations are not extra effects.")
        contrast = requested.question.model_copy(deep=True)
        contrast.prerequisites = list(dict.fromkeys([*contrast.prerequisites, "decision:target"]))
        effect = EffectRequest(id=requested.id, priority=requested.priority, contrast=contrast,
            treatment_ids=args.treatment_ids, estimand=args.estimand, comparison=args.comparison)
        accepted = self.admit(Treatments.model_validate(decision("Bound the exact requested contrast.", effects=[effect.model_dump(mode="json")])))
        return self.save("effect:" + args.effect_id, accepted.effects[0].model_dump(mode="json"))

    def assemble(self) -> tuple[Target, Treatments]:
        records = self.records()
        target = self.target()
        factors = [item for name, item in sorted(records.items()) if name.startswith("factor:")]
        treatments = Treatments.model_validate(decision("The requested interventions, assignment processes and comparisons are source-bound.",
            treatments=[item["treatment"] for item in factors], values=[value for item in factors for value in item["values"]],
            assignments=[item for name, item in sorted(records.items()) if name.startswith("assignment:")],
            effects=[item for name, item in sorted(records.items()) if name.startswith("effect:")]))
        brief = self.context.analysis_brief
        if brief is None or sorted(sorted(item.variable.column_ids) for item in treatments.treatments) != sorted(sorted(item.column_ids) for item in brief.treatments):
            raise ValueError("The study must cover exactly the applied brief's treatment groupings.")
        treatments = contrasts.canonicalize(treatments, {})
        self.validate(treatments, {"target": target.model_dump(mode="json")})
        return target, treatments

    def publish(self, args: Publish, key: str) -> dict[str, Any]:
        self.store.submission(self.task, key)
        target, treatments = self.assemble()
        records = {"target": target.model_dump(mode="json"), "treatments": treatments.model_dump(mode="json"),
                   "column_inventory": sorted(self.context.columns)}
        records["review"] = require_review(self, records, "Check that the target and all factors, assignment processes, source-defined values and exact requested comparisons faithfully preserve the frozen question and source evidence. Check source entailment, joint assignment and temporal scope. Estimator selection belongs downstream.")
        return {"handoff": self.store.publish(self.task, records, args.summary, args.unresolved)}

    def tools(self) -> list[Tool]:
        return [*self.common_tools(),
            Tool("set_target", "Bind the requested target, population, unit, timing and exact effect questions. Preserve the applied brief's outcome grouping. Each source Claim selects one compatible reviewed record; other facts stay separately linked. Returns validated target or specific corrections.", TargetInput, self.set_target, claims=True),
            Tool("set_factor", "Add or revise one intervention variable and its source-defined values. One factor represents a variable, not an arm. Preserve the brief's grouping and typed literal codes. Returns the bound factor; revisions invalidate dependent provisional assignments and comparisons.", FactorInput, self.set_factor, claims=True),
            Tool("set_assignment", "Add or revise one assignment process covering already-bound factors. Preserve joint assignment, unit, timing, mechanism and history. Do not infer independent assignment. Returns the source-bound process.", AssignmentInput, self.set_assignment, claims=True),
            Tool("set_comparison", "Bind one existing requested effect to its exact treatment factors and typed comparison. The backend carries its accepted question and priority. Preserve operands, direction and requested family; do not choose an estimator here.", ComparisonInput, self.set_comparison, claims=True),
            Tool("publish_handoff", "Validate the complete study and seal it for analysis design. Requires all requested factors, assignments and comparisons. At most three submissions are permitted; inspect and fix provisional records before retrying.", Publish, self.publish),
        ]
