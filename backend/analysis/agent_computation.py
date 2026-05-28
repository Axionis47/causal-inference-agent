"""Program review, exact approval, deterministic execution and output handoff."""

import difflib
import json
from typing import Any, cast
from uuid import uuid5

from backend.analysis import code_checks, domain, numerical_api, packets
from backend.analysis.agent_context import NoArguments, Publish, Workspace
from backend.analysis.agent_review import require_review
from backend.analysis.agent_runtime import Tool
from backend.analysis.contracts import Program, Snapshot
from backend.analysis.records import approval_hash, programs, requested_revision
from backend.context.service import fingerprint
from backend.execution import runner as sandbox
from backend.storage import files


class Computation(Workspace):
    def snapshot(self) -> Snapshot:
        handoff = self.store.accepted("design")
        if handoff is None:
            raise ValueError("Computation requires the accepted design handoff.")
        return Snapshot.model_validate(handoff["records"]["snapshot"])

    def inspect(self, args: NoArguments, key: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        assert snapshot.input_projection is not None
        saved = programs(self.run)
        current = saved[-1] if saved else None
        execution = cast(dict[str, Any] | None, self.run.receipts.get(f"execution:{current['revision']}")) if current else None
        # Failed target values are never sent to programming repairs.
        return {"input_columns": [{"column_id": item.column_id, "header": item.column_id} for item in snapshot.input_projection.columns],
                "output_contract": code_checks.output_contract(snapshot.plan.model_dump(mode="json")),
                "program_summary": code_checks.program_summary(snapshot.plan.model_dump(mode="json")),
                "current_program": current,
                "review_feedback": self.run.receipts.get(f"revision_request:{current['revision']}") if current else None,
                "execution": execution if execution and execution["valid"] else None,
                "failure": {"kind": "programming" if domain.repair_error(execution) else "scientific_or_output_validation",
                            "message": execution.get("validation_error") or "The approved computation did not produce a valid result. Preserve the failure; do not change the frozen method.",
                            "target_values_withheld": True} if execution and not execution["valid"] else None,
                "programming_failure": domain.repair_error(execution) if execution else None,
                "repairs_remaining": 2 - int(current["revision"]) if current else 2}

    def propose(self, args: Program, key: str) -> dict[str, Any]:
        saved = programs(self.run)
        revision_requested = requested_revision(self.run)
        if saved and f"execution:{saved[-1]['revision']}" not in self.run.receipts and not revision_requested:
            raise ValueError("The saved program is waiting for exact approval or execution. It cannot be replaced.")
        snapshot = self.snapshot()
        science = snapshot.model_dump(mode="json")
        revision = len(saved)
        if revision > 2:
            raise ValueError("The two programming repairs have been used.")
        if saved and not revision_requested:
            execution = cast(dict[str, Any], self.run.receipts[f"execution:{saved[-1]['revision']}"])
            failure = domain.repair_error(execution)
            if failure is None:
                raise ValueError("Only a saved repairable programming failure permits changing the program.")
            domain.validate_repair(saved[-1]["program"]["code"], args.code)
        program = args.model_copy(update={"summary": code_checks.program_summary(science["plan"])})
        verification = domain.validate_program(program)
        verification["numerical_api"] = numerical_api.confidence_arguments(program.code, science["plan"])
        reviewed = require_review(self, {"snapshot": packets.science(science),
            "program": program.model_dump(mode="json"), "verification": verification, "output_contract": code_checks.output_contract(science["plan"])},
            "Independently review this exact program before target execution. Verify canonical column IDs, treatment coding and direction, all effects, population, missingness, adjustment, estimator, inference, seed, finite result schema and honest failure conditions against the frozen specification. Trace actual API argument values, not comments or variable names: scipy.stats.t.interval and norm.interval take confidence coverage, whereas ppf takes a cumulative quantile. Reject wrong probability levels, fallback estimators, fabricated values, changed methods, undeclared exclusions or result-driven choices.")
        runtime = sandbox.ready()
        record = {"revision": revision, "science_hash": fingerprint(science), "runtime_hash": runtime["runtime_hash"],
            "program": program.model_dump(mode="json"), "review": reviewed["review"], "verification": verification,
            "diff": "".join(difflib.unified_diff(saved[-1]["program"]["code"].splitlines(keepends=True), program.code.splitlines(keepends=True), fromfile="approved.py", tofile="proposed.py")) if saved else ""}
        record["hash"] = fingerprint(record)
        if self.run.snapshot is None:
            self.run.snapshot, self.run.snapshot_hash = science, record["science_hash"]
        elif self.run.snapshot != science:
            raise ValueError("A programming repair cannot change the frozen science.")
        self.run.receipts = {**self.run.receipts, f"program:{revision}": record}
        self.run.status, self.run.stage = "awaiting_approval", "computation"
        self.save(f"review:{revision}", reviewed)
        return {"status": "awaiting_approval", "revision": revision, "snapshot_hash": record["science_hash"], "program_hash": record["hash"]}

    def execute(self) -> dict[str, Any]:
        current = programs(self.run)[-1]
        snapshot = self.snapshot()
        if (fingerprint(self.run.snapshot) != self.run.snapshot_hash or current["science_hash"] != self.run.snapshot_hash
                or approval_hash(self.run) != current["hash"] or self.run.snapshot != snapshot.model_dump(mode="json")):
            raise ValueError("Execution requires exact approval of the unchanged dataset, science, code and runtime.")
        assert snapshot.input_projection is not None
        reference = f"execution:{current['revision']}"
        if reference in self.run.receipts:
            return dict(cast(dict[str, Any], self.run.receipts[reference]))
        csv, projection = domain.execution_projection(self.context,
            files().read(self.version.storage_key, snapshot.source_sha256), snapshot.input_projection)
        identity = uuid5(self.run.id, f"program:{current['hash']}")
        result = sandbox.run(identity, current["program"]["code"], csv,
            {"snapshot.json": json.dumps(self.run.snapshot).encode(), "columns.json": json.dumps(
                [{"column_id": item.column_id, "header": item.column_id} for item in snapshot.input_projection.columns]).encode()}, current["runtime_hash"])
        error = domain.execution_error(result["facts"], snapshot.frame.treatments, snapshot.plan) if result["ok"] else "Approved program failed."
        receipt = {**result, "execution_id": str(identity), "input_projection": projection,
                   "valid": bool(result["ok"] and not error), "validation_error": error}
        self.run.receipts = {**self.run.receipts, reference: receipt}
        self.store.db.commit()
        return receipt

    def publish(self, args: Publish, key: str) -> dict[str, Any]:
        self.store.submission(self.task, key)
        saved = programs(self.run)
        if not saved:
            raise ValueError("A program must be reviewed, approved and executed before publishing computation.")
        current = saved[-1]
        reference = f"execution:{current['revision']}"
        receipt = cast(dict[str, Any] | None, self.run.receipts.get(reference))
        if not receipt or not receipt["valid"]:
            raise ValueError("No validated execution exists. Inspect the failure and perform an allowed programming repair, or stop with its retained failure.")
        records = {"execution_reference": reference, "execution": receipt, "program_hash": current["hash"],
                   "snapshot_hash": self.run.snapshot_hash, "review": self.records()[f"review:{current['revision']}"]}
        return {"handoff": self.store.publish(self.task, records, args.summary, args.unresolved)}

    def tools(self) -> list[Tool]:
        return [*self.common_tools(),
            Tool("inspect_computation", "Inspect the canonical input headers, exact required numerical output keys, current program and validated output. Failed numerical values are withheld from repairs.", NoArguments, self.inspect),
            Tool("submit_program", "Submit complete deterministic Python for the frozen analysis. Static checks and independent code review run before this exact program is presented for approval. No target data is accessed. After a saved programming failure, at most two minimal repairs are allowed and each requires fresh approval.", Program, self.propose),
            Tool("publish_handoff", "Validate and hand off the approved execution receipt, computed facts and artifacts. Requires successful deterministic output validation. Reporting must use these exact saved numbers.", Publish, self.publish),
        ]
