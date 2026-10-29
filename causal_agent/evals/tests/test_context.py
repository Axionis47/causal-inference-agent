"""Structural real-model evidence grading, without making a model call."""

from types import SimpleNamespace as Run

import pytest

from causal_agent.evals.context import verify_result, verify_runs
from causal_agent.memory.context.contracts import Assessment, ExtractedContext
from causal_agent.memory.context.tests.builders import COLUMNS, SOURCE, draft


def test_result_grader_rejects_invented_citation_and_missing_column():
    result = ExtractedContext(content=draft(), assessment=Assessment(), model_calls=2)
    verify_result(result, SOURCE, COLUMNS)
    result.content.columns[1].statements[0].citations[0].quote = "not in the source"
    with pytest.raises(ValueError, match="Citation"):
        verify_result(result, SOURCE, COLUMNS)
    result.content.columns.pop()
    with pytest.raises(ValueError, match="column"):
        verify_result(result, SOURCE, COLUMNS)


def run(id, name, parent="root", **extra):
    return Run(id=id, name=name, parent_run_id=parent, end_time=True, error=None, outputs={"passed": True}, **extra)


def test_trace_grader_requires_complete_parentage_calls_and_final_gates():
    runs = [
        run("root", "acceptance:context", None),
        run("fill", "context:fill"),
        run("assess", "context:assess"),
        run("v", "context:validate"),
        run("a", "context:assessment"),
    ]
    verify_runs(runs, "root", 2)
    runs[-1].outputs["passed"] = False
    with pytest.raises(ValueError, match="Final gate"):
        verify_runs(runs, "root", 2)
    runs[-1].outputs["passed"] = True
    runs[-1].parent_run_id = "missing"
    with pytest.raises(ValueError, match="hierarchy"):
        verify_runs(runs, "root", 2)
    runs[-1].parent_run_id = "root"
    runs[-1].end_time = None
    with pytest.raises(ValueError, match="unfinished"):
        verify_runs(runs, "root", 2)
