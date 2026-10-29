"""Every built lane's ladder is declared once and the declaration is held to the code: the compiled graph has exactly the edges
rungs.yaml describes, a node jumps only where its rung says, the space's deciders are rungs, the tools are the harness's, the
budgets the judgements read are declared, and the generated tables under docs/ladders say what the declarations say."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from pathlib import Path

import pytest

from causal_agent.evals import ladder as EL
from causal_agent.explore.tools import CodeTools
from causal_agent.lane import rungs as RG
from causal_agent.lane.tools import NAMES

LANES = EL.built()
TOOL_NAMES = {*NAMES, *CodeTools(csv=Path(), out_dir=Path(), timeout=0, stdout_tail=0).names()}


def _modules(family: str):
    lane = f"causal_agent.families.{family}.lane"
    return importlib.import_module(f"{lane}.graph"), importlib.import_module(f"{lane}.nodes"), importlib.import_module(f"{lane}.knowledge")


@pytest.mark.parametrize("family", LANES)
def test_the_compiled_graph_is_the_declaration(family):
    graph, _, knowledge = _modules(family)
    spec: RG.LadderSpec = knowledge.SPEC
    drawn = graph.build().compile(name=f"test:{family}").get_graph()
    assert set(drawn.nodes) == set(spec.names) | {RG.STOP, *RG.TAIL, "__start__", "__end__"}
    edges = {(e.source, e.target) for e in drawn.edges}
    assert edges == spec.edges(), {"drawn, not declared": edges - spec.edges(), "declared, not drawn": spec.edges() - edges}


def _jumps(fn) -> set[str]:
    """Every node name a function can jump to: Command(goto="x"), Command(goto=[Send("x", ...)]) and Send("x", ...)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "Send" and node.args and isinstance(node.args[0], ast.Constant):
                out.add(str(node.args[0].value))
            if name == "Command":
                for kw in node.keywords:
                    if kw.arg == "goto" and isinstance(kw.value, ast.Constant):
                        out.add(str(kw.value.value))
    return out


@pytest.mark.parametrize("family", LANES)
def test_a_node_jumps_only_where_its_rung_says(family):
    _, nodes, knowledge = _modules(family)
    spec: RG.LadderSpec = knowledge.SPEC
    for r in spec.rungs:
        fn = getattr(nodes, r.name, None)
        if fn is None or not inspect.isfunction(fn):
            continue
        allowed = set(spec.exits_of(r)) | ({RG.STOP} if r.stop else set())
        jumps = _jumps(fn)
        assert jumps <= allowed, f"{family}.{r.name} jumps to {sorted(jumps - allowed)}, which its rung does not declare"


@pytest.mark.parametrize("family", LANES)
def test_the_declaration_names_what_the_code_has(family):
    _, nodes, knowledge = _modules(family)
    spec: RG.LadderSpec = knowledge.SPEC
    for r in spec.rungs:
        assert callable(getattr(nodes, r.name, None)), f"{family}: the rung {r.name} has no node function"
        if r.route:
            assert callable(getattr(nodes, r.route, None)), f"{family}: the route {r.route} is not a function in nodes"
        assert set(r.tools) <= TOOL_NAMES, f"{family}.{r.name} declares tools the harness lacks: {set(r.tools) - TOOL_NAMES}"
    space = getattr(knowledge, "SPACE", None)
    if space is not None:
        assert {d.decided_by for d in space.dimensions.values()} <= set(spec.names)
    for rung in spec.judgements():
        if rung.tools:
            assert _reads_declared_budget(getattr(nodes, rung.name)), f"{family}.{rung.name}: tool episode does not read its declared budget"


def _reads_declared_budget(fn, seen=None) -> bool:
    """Follow local episode helpers rather than require one helper spelling."""
    seen = set() if seen is None else seen
    if fn in seen:
        return False
    seen.add(fn)
    module = inspect.getmodule(fn)
    calls = [n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))) if isinstance(n, ast.Call)]
    for call in calls:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "budget":
            return True
        if isinstance(call.func, ast.Name):
            child = getattr(module, call.func.id, None)
            if inspect.isfunction(child) and inspect.getmodule(child) == module and _reads_declared_budget(child, seen):
                return True
    return False


@pytest.mark.parametrize("family", LANES)
def test_each_executable_endpoint_has_a_reachable_freeze_and_engine(family):
    from causal_agent.families import registry as R
    from causal_agent.lane.endpoints import verify_catalogue

    _, _, knowledge = _modules(family)
    catalogue = R.needs()[family].endpoints
    assert catalogue is not None
    assert not verify_catalogue(catalogue, knowledge.SPEC, {e.name for e in knowledge.load_estimators()})
    edges = knowledge.SPEC.edges()

    def reachable(start, end):
        pending = [start]
        visited = set()
        while pending:
            node = pending.pop()
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)
            pending.extend(dst for src, dst in edges if src == node)
        return False

    for endpoint in catalogue.endpoints:
        if not endpoint.executable:
            continue
        stages = ["__start__", *endpoint.trajectory]
        assert all(reachable(a, b) for a, b in zip(stages, stages[1:])), endpoint.id
        assert knowledge.SPEC.goal in endpoint.trajectory
    # An endpoint may have only declared sensitivity or assumption diagnostics. A falsification is not mandatory.
    assert len({e.id for e in catalogue.endpoints}) == len(catalogue.endpoints)


def test_the_ladder_tables_say_what_the_declarations_say():
    assert EL.stale() == [], "run make ladder"
