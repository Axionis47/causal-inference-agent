"""Evidence-grounded explanation of saved computation for results and follow-up chat."""

from typing import Any

from pydantic import Field

from backend import numerical
from backend.analysis import domain
from backend.analysis.agent_context import NoArguments, Publish, Workspace, decision, require_checks
from backend.analysis.agent_review import require_review
from backend.analysis.agent_runtime import Tool
from backend.analysis.contracts import AnalysisReport, Conclusion, Result
from backend.models import now
from backend.responses import Reply
from backend.validation import Contract


class ReportInput(Contract):
    report: AnalysisReport
    citations: list[str] = Field(min_length=1, max_length=60)
    limitations: list[str] = Field(default_factory=list, max_length=20)


class Explanation(Workspace):
    def inputs(self) -> dict[str, Any]:
        design, computation = self.store.accepted("design"), self.store.accepted("computation")
        if design is None or computation is None:
            raise ValueError("Explanation requires accepted design and computation handoffs.")
        execution = computation["records"]["execution"]
        if not execution["valid"]:
            raise ValueError("A successful report requires validated saved execution.")
        return {"snapshot": design["records"]["snapshot"], "execution": execution,
                "numeric_evidence": {**{f"execution.{key}": {"basis": "computed", "value": value} for key, value in execution["facts"].items()},
                                     "question": {"basis": "prespecified_user_request", "value": self.run.question}}}

    def inspect(self, args: NoArguments, key: str) -> dict[str, Any]:
        data = self.inputs()
        return {"numeric_evidence": {name: {**record, "supported_values": numerical.source_values(record["value"])}
                                     for name, record in data["numeric_evidence"].items()},
                "citation_handles": ["question", "execution", "decision:plan", "decision:causal"],
                "citation_guidance": "Use these exact handles or frozen source references. Handoff hashes and execution:<revision> are identities, not citation handles. Cite question for prespecified confidence levels and treatment codes. Every digit-written number, including confidence levels, needs numbers metadata; exact cited question literals are bound by the backend.",
                "filenames": sorted(data["execution"]["files"]),
                "assumptions": data["snapshot"]["frame"]["causal"]["assumptions"], "limitations": data["snapshot"]["plan"]["limitations"]}

    def set_report(self, args: ReportInput, key: str) -> dict[str, Any]:
        data = self.inputs()
        reply = Reply.model_validate(args.report.model_dump(mode="json"))
        if "question" in args.citations:
            reply = numerical.bind_literals(reply, {"question": self.run.question})
        numerical.validate(reply, data["numeric_evidence"])
        prior = {**data["snapshot"]["frame"], "plan": data["snapshot"]["plan"]}
        ledger = {entry["id"]: entry for record in prior.values() if isinstance(record, dict) for entry in record.get("assumptions", [])}
        conclusion = Conclusion.model_validate(decision("Grounded explanation of the saved approved computation.",
            citations=args.citations, assumptions=list(ledger.values()), unresolved=args.limitations,
            report={"text": reply.text, "numbers": [item.model_dump(mode="json") for item in reply.numbers]}))
        require_checks(domain.checks(conclusion, self.context, prior, {"execution", *data["snapshot"]["diagnostic_refs"]}, scientific_version=19))
        return self.save("conclusion", conclusion.model_dump(mode="json"))

    def publish(self, args: Publish, key: str) -> dict[str, Any]:
        self.store.submission(self.task, key)
        saved = self.records().get("conclusion")
        if saved is None:
            raise ValueError("Submit a numerically grounded report first.")
        data = self.inputs()
        reviewed = require_review(self, {"conclusion": saved, "numeric_evidence": data["numeric_evidence"]},
            "Review the final explanation against the accepted design and computed evidence. Every numerical statement must have the exact saved source; maintain conditional causal assumptions, population, contrast direction, uncertainty and limitations. Do not allow a synthetic example to be represented as an empirical generalization.")
        conclusion = Conclusion.model_validate(saved)
        execution = data["execution"]
        result = Result(report=Reply.model_validate(conclusion.report.model_dump(mode="json")), facts=execution["facts"],
            filenames=sorted(execution["files"]), execution_ok=True, stdout=execution["stdout"], stderr=execution["stderr"])
        handoff = self.store.publish(self.task, {"conclusion": saved, "review": reviewed}, args.summary, args.unresolved)
        self.run.result, self.run.status, self.run.error, self.run.updated_at = result.model_dump(mode="json"), "succeeded", None, now()
        return {"handoff": handoff}

    def tools(self) -> list[Tool]:
        return [*self.common_tools(),
            Tool("inspect_results", "Inspect exact saved numerical evidence handles, filenames and frozen assumptions. Cite numerical sources using these keys and supported values; never estimate numbers from charts.", NoArguments, self.inspect),
            Tool("set_report", "Draft the explanation using computed evidence and the frozen design. Cite execution, decision:plan, decision:causal or exact source references. Explicitly distinguish conditional assumptions and limitations. Every number must resolve to a saved numerical source or the cited immutable question.", ReportInput, self.set_report),
            Tool("publish_handoff", "Obtain an independent report check and publish the grounded saved results for reopening and follow-up chat. At most three submissions.", Publish, self.publish),
        ]
