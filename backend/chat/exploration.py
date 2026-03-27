"""Bounded open-ended visualization with inspectable computational evidence."""

import csv
import io
import json
import math
import re
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid5

from PIL import Image
from pydantic import BaseModel

from backend.chat.contracts import Exploration, ExploreAction
from backend.execution import runner as sandbox
from backend.storage import files

Ask = Callable[[str, type[BaseModel], str, dict[str, object], list[bytes]], BaseModel]
Remember = Callable[[str, Any], None]


def preview(name: str, raw: bytes) -> tuple[str, list[bytes]]:
    if name.endswith(".png"):
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("The image exceeds the inspection pixel limit.")
            image.load()
            image.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return f"{name}: image pixels supplied for inspection.", [buffer.getvalue()]
    if name.endswith(".csv"):
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        from itertools import islice

        return f"{name}: preview only, first 20 rows including header: " + json.dumps(
            list(islice(reader, 20)), ensure_ascii=False
        )[:16000], []
    return f"{name}: text preview (up to 16000 characters): " + raw.decode("utf-8")[:16000], []


def validate_record(record: Exploration, steps: list[dict[str, Any]], inspected: set[str]) -> None:
    if record.status == "needs_input":
        if not record.needs or record.step or record.figure or record.table or record.facts:
            raise ValueError("needs_input requires specific needs, step=0 and no products.")
        return
    if record.step < 1 or record.step > len(steps) or not steps[record.step - 1]["ok"]:
        raise ValueError("Select a recorded successful script step.")
    step = steps[record.step - 1]
    if not record.method.strip() or not record.findings or not record.limitations:
        raise ValueError("Include method, supported findings and limitations.")
    if not (record.facts or record.figure or record.table):
        raise ValueError("Select at least one existing product.")
    if any(name not in step["facts"] or name in {"figure", "table"} for name in record.facts):
        raise ValueError("Select actual non-reserved fact names from the chosen step.")
    if record.figure and f"step_{record.step}/figure.png" not in inspected:
        raise ValueError("Inspect the selected figure pixels before delivering it.")
    if record.table:
        if (
            record.table not in step["files"]
            or record.table == "facts.json"
            or not record.table.endswith((".csv", ".json"))
        ):
            raise ValueError("Select an existing CSV/JSON summary table, not facts.json.")
        meta = step["files"][record.table]
        raw = files().read(meta["key"], meta["sha256"])
        if record.table.endswith(".json"):
            if not json.loads(raw):
                raise ValueError("The selected table is empty.")
        elif len(list(csv.reader(io.StringIO(raw.decode("utf-8"))))) < 2:
            raise ValueError("The selected CSV needs a header and data.")
    allowed = (
        set(record.facts) | ({"figure"} if record.figure else set()) | ({"table"} if record.table else set())
    )
    for finding in record.findings:
        citations = re.findall(r"\[([^\]]+)\]", finding)
        if not citations or not set(citations) <= allowed:
            raise ValueError(
                "Each finding needs exact separate bracket citations to selected facts, [figure] or [table]."
            )
    values = [step["facts"][key] for key in record.facts]
    # This verifies numerical support, not the scientific validity of the chosen computation.
    for text in [record.summary, *record.findings]:
        without_cites = re.sub(r"\[[^\]]+\]", "", text)
        for token in re.findall(r"(?<![\w])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", without_cites):
            number = float(token)
            places = len(token.split(".")[1]) if "." in token and "e" not in token.lower() else 0
            if not any(
                math.isclose(number, value, rel_tol=1e-8, abs_tol=1e-9) or number == round(value, places)
                for value in values
            ):
                raise ValueError(
                    f"The number {token} lacks support in the selected numeric facts. Include its computed fact or remove the numerical claim."
                )


def explore(
    identity: UUID,
    request: str,
    context: dict[str, object],
    csv_bytes: bytes,
    supplied: dict[str, bytes],
    prompts: dict[str, object],
    receipts: dict[str, Any],
    ask: Ask,
    remember: Remember,
) -> dict[str, Any]:
    if "exploration" in receipts:
        return dict(receipts["exploration"])
    if "runtime" not in receipts:
        remember("runtime", sandbox.ready())
    runtime_hash = receipts["runtime"]["runtime_hash"]
    inputs = {"dataset.csv": csv_bytes, **supplied}
    trail: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    inspected: set[str] = set()
    media: list[bytes] = []
    inspections = 0
    final_attempts = 0
    for index in range(21):
        payload: dict[str, object] = {
            "task": request,
            "context": context,
            "inputs": list(inputs),
            "history": trail,
            "scripts_remaining": 8 - len(steps),
            "inspections_remaining": 16 - inspections,
        }
        action = ExploreAction.model_validate(
            ask(f"action:{index}", ExploreAction, str(prompts["explore"]), payload, media).model_dump()
        )
        media = []
        if action.action == "finish":
            final_attempts += 1
            if final_attempts > 3:
                raise ValueError("The visualization did not pass its three final validation attempts.")
            record = Exploration.model_validate(
                ask(
                    f"finish:{index}",
                    Exploration,
                    str(prompts["finish"]),
                    payload,
                    [],
                ).model_dump()
            )
            try:
                validate_record(record, steps, inspected)
            except ValueError as error:
                trail.append({"rejected_record": record.model_dump(), "validation_error": str(error)})
                continue
            selected = steps[record.step - 1] if record.status == "completed" else None
            result = {
                "exploration": record.model_dump(mode="json"),
                "facts": {key: selected["facts"][key] for key in record.facts} if selected else {},
                "files": selected["files"] if selected else {},
                "runtime_hash": runtime_hash,
                "scripts": len(steps),
                "inspections": inspections,
            }
            remember("exploration", result)
            return result
        key = f"tool:{index}"
        if key not in receipts:
            try:
                if action.action == "run_code":
                    if len(steps) >= 8:
                        raise ValueError(
                            "The eight-script budget is exhausted. Use existing evidence or report the failure."
                        )
                    result = sandbox.run(uuid5(identity, key), action.code, csv_bytes, inputs, runtime_hash)
                else:
                    if inspections >= 16:
                        raise ValueError("The sixteen-inspection budget is exhausted.")
                    if not action.reference and action.action == "inspect_input":
                        result = {"text": json.dumps(list(inputs)), "reference": ""}
                    elif action.reference not in inputs or (
                        action.action == "inspect_output"
                    ) != action.reference.startswith("step_"):
                        raise ValueError(
                            "Select a filename from the supplied inputs or a successful step_N output."
                        )
                    else:
                        text, _ = preview(action.reference, inputs[action.reference])
                        result = {"text": text, "reference": action.reference}
                remember(key, result)
            except ValueError as error:
                remember(key, {"error": str(error)})
        result = receipts[key]
        if "error" in result:
            trail.append({"action": action.action, "error": result["error"]})
            continue
        if action.action == "run_code":
            steps.append(result)
            number = len(steps)
            trail.append(
                {
                    "step": number,
                    "code": action.code,
                    "ok": result["ok"],
                    "stdout_tail": result["stdout"],
                    "stderr_tail": result["stderr"],
                    "facts": result["facts"],
                    "files": list(result["files"]),
                }
            )
            if result["ok"]:
                for name, meta in result["files"].items():
                    if name != "code.py":
                        inputs[f"step_{number}/{name}"] = files().read(meta["key"], meta["sha256"])
                reference = f"step_{number}/figure.png"
                if reference in inputs and inspections < 16:
                    try:
                        text, media = preview(reference, inputs[reference])
                        inspections += 1
                        inspected.add(reference)
                        trail.append({"inspection": text})
                    except (OSError, ValueError) as error:
                        trail.append({"inspection_error": str(error)})
        else:
            inspections += 1
            reference = result["reference"]
            trail.append({"inspection": result["text"]})
            if reference:
                _, media = preview(reference, inputs[reference])
                inspected.add(reference)
    raise ValueError("Visualization reached its bounded action limit without a validated result.")
