"""The assembly tool registry the ReAct loop offers.

Read tools (list_files, inspect_join_keys) return computed facts about the
bundle so the model's choices are grounded; propose tools (propose_join,
propose_concat) validate against the files' columns and record the choice on
a ledger; finish_assembly freezes the base file. Tools are offered only when
the bundle has more than one file, so single-file datasets never invoke the
loop. Handlers mutate only the ledger, never run state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from src.analysis_v2.agents.base import LoopTool
from src.analysis_v2.spec import AssemblyJoin, AssemblyPlan, AssemblyToolCall, AssemblyToolTrace
from src.domain.dataset_manifest import DatasetManifest

# runner.assembly is imported lazily inside the functions below: importing it at
# module load would cycle (registry -> assembly_planner -> tools -> runner ->
# registry). By call time the runner package is fully initialized.

_NONE = {"type": "object", "properties": {}}
_INSPECT = {
    "type": "object",
    "properties": {"left_file": {"type": "string"}, "right_file": {"type": "string"}},
    "required": ["left_file", "right_file"],
}
_JOIN = {
    "type": "object",
    "properties": {
        "right_file": {"type": "string"},
        "on": {"type": "array", "items": {"type": "string"}},
        "how": {"type": "string", "enum": ["left", "inner"]},
    },
    "required": ["right_file", "on"],
}
_CONCAT = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {"type": "string"}}},
    "required": ["files"],
}
_FINISH = {
    "type": "object",
    "properties": {"base_file": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["base_file"],
}


@dataclass
class AssemblyLedger:
    winner: str
    base_file: str = ""
    concat_files: list[str] = field(default_factory=list)
    joins: list[AssemblyJoin] = field(default_factory=list)
    rationale: str = ""
    calls: list[AssemblyToolCall] = field(default_factory=list)

    def running_columns(self, manifest: DatasetManifest, job_id: str) -> set[str]:
        """Columns the running frame would have so far: base + concats + prior
        joins. Used to validate that a join key exists on the left side."""
        base = self.base_file or self.winner
        cols: set[str] = set(_columns(manifest, job_id, base))
        for name in self.concat_files:
            cols |= set(_columns(manifest, job_id, name))
        for join in self.joins:
            cols |= set(_columns(manifest, job_id, join.right_file))
        return cols


def _names(manifest: DatasetManifest) -> set[str]:
    return {f.name for f in manifest.files}


def _columns(manifest: DatasetManifest, job_id: str, name: str) -> list[str]:
    """The file's columns from the manifest, or read from disk if absent."""
    from src.analysis_v2.runner.assembly import DatasetUnavailable, _read_manifest_file

    entry = next((f for f in manifest.files if f.name == name), None)
    if entry is not None and entry.columns:
        return list(entry.columns)
    try:
        return list(_read_manifest_file(job_id, manifest, name).columns)
    except DatasetUnavailable:
        return []


def build_assembly_tools(
    manifest: DatasetManifest, job_id: str
) -> tuple[AssemblyLedger, list[LoopTool]]:
    """(ledger, tools). Tools are offered only for a multi-file bundle; a
    single-file bundle returns no tools so the agent uses today's behavior."""
    from src.analysis_v2.runner.assembly import DatasetUnavailable, _read_manifest_file

    ledger = AssemblyLedger(winner=manifest.winner or "")
    if len(manifest.files) <= 1:
        return ledger, []

    def list_files() -> str:
        ledger.calls.append(AssemblyToolCall(name="list_files", kind="read", status="ok"))
        lines = []
        for f in manifest.files:
            flag = " [current default]" if f.name == manifest.winner else ""
            cols = ", ".join((f.columns or _columns(manifest, job_id, f.name))[:25])
            lines.append(f"{f.name}{flag}: {f.n_rows if f.n_rows is not None else '?'} "
                         f"rows; columns: {cols or 'unknown'}")
        return "Files in the bundle:\n" + "\n".join(lines)

    def inspect_join_keys(left_file: str, right_file: str) -> str:
        bad = [n for n in (left_file, right_file) if n not in _names(manifest)]
        if bad:
            return _reject(ledger, "inspect_join_keys", "read",
                           {"left_file": left_file, "right_file": right_file},
                           f"unknown file(s): {bad}")
        try:
            left = _read_manifest_file(job_id, manifest, left_file)
            right = _read_manifest_file(job_id, manifest, right_file)
        except DatasetUnavailable as exc:
            return _reject(ledger, "inspect_join_keys", "read",
                           {"left_file": left_file, "right_file": right_file}, str(exc))
        shared = [c for c in left.columns if c in right.columns]
        ledger.calls.append(AssemblyToolCall(
            name="inspect_join_keys", kind="read", status="ok",
            args={"left_file": left_file, "right_file": right_file}))
        if not shared:
            return f"no shared columns between {left_file} and {right_file}"
        facts = [
            f"{c}: unique_in_left={left[c].is_unique}, "
            f"unique_in_right={right[c].is_unique}, "
            f"shared_values={len(set(left[c].dropna()) & set(right[c].dropna()))}"
            for c in shared[:10]
        ]
        return "shared columns (candidate join keys):\n" + "\n".join(facts)

    def propose_join(right_file: str, on: list[str], how: str = "left") -> str:
        how = how if how in ("left", "inner") else "left"
        args = {"right_file": right_file, "on": on, "how": how}
        if right_file not in _names(manifest):
            return _reject(ledger, "propose_join", "join", args, f"unknown file {right_file}")
        right_cols = set(_columns(manifest, job_id, right_file))
        left_cols = ledger.running_columns(manifest, job_id)
        missing_right = [c for c in on if c not in right_cols]
        missing_left = [c for c in on if c not in left_cols]
        if missing_right or missing_left:
            return _reject(ledger, "propose_join", "join", args,
                           f"key(s) not found: left={missing_left} right={missing_right}")
        ledger.joins.append(AssemblyJoin(
            right_file=right_file, on=list(on),
            how=cast(Literal["left", "inner"], how)))
        ledger.calls.append(AssemblyToolCall(name="propose_join", kind="join",
                                             status="ok", args=args))
        return f"join recorded: {how} join {right_file} on {on}"

    def propose_concat(files: list[str]) -> str:
        args = {"files": files}
        bad = [f for f in files if f not in _names(manifest)]
        if bad:
            return _reject(ledger, "propose_concat", "concat", args, f"unknown file(s): {bad}")
        base_cols = set(_columns(manifest, job_id, ledger.base_file or ledger.winner))
        mismatched = [f for f in files if set(_columns(manifest, job_id, f)) != base_cols]
        if mismatched:
            return _reject(ledger, "propose_concat", "concat", args,
                           f"not the same schema as the base: {mismatched}")
        ledger.concat_files.extend(files)
        ledger.calls.append(AssemblyToolCall(name="propose_concat", kind="concat",
                                             status="ok", args=args))
        return f"concat recorded: {files}"

    def finish_assembly(base_file: str, rationale: str = "") -> str:
        args = {"base_file": base_file}
        if base_file not in _names(manifest):
            return _reject(ledger, "finish_assembly", "finish", args,
                           f"unknown base file {base_file}")
        ledger.base_file = base_file
        ledger.rationale = rationale[:1000]
        ledger.calls.append(AssemblyToolCall(name="finish_assembly", kind="finish",
                                             status="ok", args=args))
        return f"assembly finished: base {base_file}"

    tools = [
        LoopTool("list_files",
                 "List every file in the bundle with its row count and columns.",
                 _NONE, list_files),
        LoopTool("inspect_join_keys",
                 "Report shared columns and their uniqueness/overlap between two files.",
                 _INSPECT, inspect_join_keys),
        LoopTool("propose_join",
                 "Merge a sibling file onto the running frame on a shared key (how: left|inner).",
                 _JOIN, propose_join),
        LoopTool("propose_concat",
                 "Stack same-schema shard files onto the base file (vertical concat).",
                 _CONCAT, propose_concat),
        LoopTool("finish_assembly",
                 "Freeze the plan: set the base file and a one-line rationale.",
                 _FINISH, finish_assembly),
    ]
    return ledger, tools


def _reject(ledger: AssemblyLedger, name: str, kind: str, args: dict[str, Any], reason: str) -> str:
    ledger.calls.append(AssemblyToolCall(name=name, kind=kind, args=args,
                                         status="rejected", rejected=reason))
    return f"rejected: {reason}"


def collect_plan(ledger: AssemblyLedger, manifest: DatasetManifest) -> AssemblyPlan:
    """Freeze the ledger into a plan; an empty ledger is the single-winner default."""
    return AssemblyPlan(
        base_file=ledger.base_file or manifest.winner or "",
        concat_files=list(ledger.concat_files),
        joins=list(ledger.joins),
        rationale=ledger.rationale,
    )


def collect_trace(ledger: AssemblyLedger, *, exhausted: bool = False) -> AssemblyToolTrace:
    return AssemblyToolTrace(calls=list(ledger.calls), exhausted=exhausted)
