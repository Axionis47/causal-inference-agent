"""Controlled analysis setup, distinct from the separately tested chat Apply journey.

An empty initial revision is seeded with the existing persistence service. All
source statements and the brief are then applied through the real context API
before the conversation is created, preserving immutable conversation identity.
"""

from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from backend.context.contracts import empty_content
from backend.context.service import append_revision, locked_pack
from backend.db import engine
from backend.models import ContextRevision, Dataset, DatasetVersion

QUESTION = "Randomized group effect on score in the study sample"


def ready_revision(owner, version_id, expected_revision_id=None, *, question=QUESTION, treatment_names=("group",)):
    version_id = UUID(str(version_id))
    with Session(engine()) as db:
        version = db.get(DatasetVersion, version_id)
        dataset = db.get(Dataset, version.dataset_id)
        pack = locked_pack(db, version_id)
        parent_id = expected_revision_id or pack.current_revision_id
        if parent_id:
            parent = db.get(ContextRevision, UUID(str(parent_id)))
        else:
            parent = append_revision(db, pack, None, uuid4(), uuid4(), "f" * 64, dataset.owner_id,
                                     empty_content(version_id, version.columns), [], "reviewed")
        db.commit()
        revision_id = str(parent.id)
    base = f"/api/versions/{version_id}/context"
    content = owner.get(base + "/revisions/" + revision_id).json()["content"]
    ids = {column["identity"]["name"]: cid for cid, column in content["columns"].items()}
    assert {"score", *treatment_names} <= ids.keys(), "This controlled fixture binds explicitly named test variables."
    edits = []

    def add(key, value, **statement):
        index = len(edits)
        edits.append({"action": "add", "key": key, "statement": {"value": value, **statement},
                      "reason": "Explicit controlled test-study account, applied before analysis."})
        return {"source_keys": [f"edit:{index}"]}

    meanings = {}
    for name, cid in ids.items():
        value = "Randomized intervention assignment; a denotes the control category and b the active category." if name == "group" else "Second randomized intervention assignment; numeric 0 denotes control and numeric 1 denotes active." if name == "second" and name in treatment_names else "Recorded follow-up score." if name == "score" else "Recorded optional study attribute."
        meanings[name] = add(f"columns.{cid}.definition", value)
        add(f"columns.{cid}.collection_time", "Assigned at baseline." if name in treatment_names else "Measured after group assignment." if name == "score" else "Measured before group assignment.")
    comparison = add("dataset.question", question + ". Compare mean score under category b minus category a." + (" Separately compare numeric second=1 minus second=0." if "second" in treatment_names else ""))
    population = add("dataset.population", "Enrolled study participants.")
    unit = add("dataset.observations", "Each row is one enrolled participant.")
    timing = add("dataset.measurement", "The score is recorded after the baseline intervention assignment.")
    change = add("dataset.study_process", "Group was randomly assigned at baseline; a is the control category and b the active category.")
    changes = {"group": change}
    if "second" in treatment_names:
        changes["second"] = add("dataset.study_process", "Second was randomly assigned at baseline; numeric 0 is the control category and numeric 1 the active category.", column_ids=[ids["second"]])
    body = {"submission_id": str(uuid4()), "expected_revision_id": revision_id, "state": "reviewed", "edits": edits,
            "analysis_brief": {"question": question, "target": {"column_ids": [ids["score"]], "meaning": meanings["score"]},
                               "treatments": [{"id": name + "_assignment", "column_ids": [ids[name]], "meaning": meanings[name], "change": changes[name]} for name in treatment_names],
                               "comparison": comparison, "population": population, "unit": unit, "timing": timing}}
    response = owner.post(base + "/revisions", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def create(owner):
    data = owner.post("/api/datasets", data={"title": "Chat sample", "submission_id": str(uuid4())},
                      files={"file": ("sample.csv", b"group,score\na,2\na,4\nb,8\n", "text/csv")}).json()
    version_id = data["version"]["id"]
    revision = ready_revision(owner, version_id)
    body = {"title": "Discuss scores", "submission_id": str(uuid4()), "context_revision_id": revision["id"]}
    base = f"/api/versions/{version_id}/conversations"
    response = owner.post(base, json=body)
    assert response.status_code == 201, response.text
    return base, body, response.json()


def provider_proposal(schema, payload, value):
    """Controlled provider proposals, never final claim admission or fake source authority."""
    from historical_analysis.work import ColumnScreen

    from backend.analysis.contracts import CausalAccount, Roles, Target, Treatments

    raw = value.model_dump(mode="json")

    def proposed(text, *, rule="source", references=None, obligation=None):
        return {"statement": text, "rule": rule, "references": references or [], "prerequisites": [], "required_obligation": obligation}

    def source(prefix):
        record = next((item for item in payload["evidence_records"] if item["statement"]["value"].startswith(prefix)), None)
        assert record is not None, f"Controlled fixture source is absent: {prefix}"
        return proposed(record["statement"]["value"], references=[record["key"]])

    def definition(identity):
        column = next(item for item in payload["column_identities"] if item["column_id"] == identity)
        prefix = "Recorded follow-up score." if column["name"] == "score" else "Randomized intervention assignment;" if column["name"] == "group" else "Second randomized intervention assignment;" if column["name"] == "second" else "Recorded optional study attribute."
        return source(prefix)

    def meaning(variable):
        variable["meaning"] = definition(variable["id"])

    def assignment_source(identity):
        column = next(item for item in payload["column_identities"] if item["column_id"] == identity)
        return source("Second was randomly assigned" if column["name"] == "second" else "Group was randomly assigned")

    request = proposed(payload["question"], references=["question"])
    if schema is ColumnScreen:
        meaning(raw["variable"])
    elif schema is Target:
        meaning(raw["outcome"])
        raw.update(measurement=source("The score is recorded"), population=source("Enrolled study participants."),
                   unit=source("Each row is one enrolled"), timing=source("The score is recorded"))
        for effect in raw["requested_effects"]:
            effect["question"] = request
    elif schema is Treatments:
        for treatment in raw["treatments"]:
            meaning(treatment["variable"])
            treatment.update(change=definition(treatment["variable"]["id"]), timing=assignment_source(treatment["variable"]["id"]), assignment=None)
        for level in raw["values"]:
            level["meaning"] = definition(level["variable_id"])
        for process in raw["assignments"]:
            process.update(unit=source("Each row is one enrolled"), timing=assignment_source(process["treatment_ids"][0]), mechanism=assignment_source(process["treatment_ids"][0]))
        for effect in raw["effects"]:
            effect.pop("contrast")
            effect.pop("priority")
            effect["comparison"]["evidence"] = request
    elif schema is Roles:
        for role in raw["assignments"]:
            role.update(variable={"id": role["variable"]["id"]}, roles=["unknown"], timing="unknown",
                        account=proposed("The optional scientific role has not been established.", rule="unknown"),
                        timing_evidence=proposed("No admitted event-relative timing is asserted.", rule="unknown"),
                        reason="Unresolved optional interpretation is retained without scientific authority.",
                        time_scope="The task's supplied event scope; timing remains unknown.")
    elif schema is CausalAccount:
        for edge in raw["edges"]:
            edge.pop("basis")
            edge.update(rule="working_assumption", references=[], prerequisites=[], required_obligation="graph_structure")
        for identification in raw["identifications"]:
            effect = next(item for item in payload["decisions"]["treatments"]["effects"] if item["id"] == identification["effect_id"])
            identification["evidence"] = proposed(
                "The documented randomized assignment identifies this fixed contrast, conditional on the study account and consistency.",
                rule="working_assumption", references=assignment_source(effect["treatment_ids"][0])["references"], obligation="source_assignment",
            )
    return raw
