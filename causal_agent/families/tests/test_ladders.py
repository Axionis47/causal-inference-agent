"""Every built lane's ladder is declared once and the declaration is held to the code: the compiled graph has exactly the edges
rungs.yaml describes, a node jumps only where its rung says, the space's deciders are rungs, the tools are the harness's, the
budgets the judgements read are declared, and the generated tables under docs/ladders say what the declarations say."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap

import pytest

from causal_agent.evals import ladder as EL
from causal_agent.lane import rungs as RG
from causal_agent.lane.tools import NAMES

LANES = EL.built()


def _modules(family: str):
    lane = f"causal_agent.families.{family}.lane"
    return importlib.import_module(f"{lane}.graph"), importlib.import_module(f"{lane}.nodes"), importlib.import_module(f"{lane}.knowledge")


@pytest.mark.parametrize("family", LANES)
def test_the_compiled_graph_is_the_declaration(family):
    graph, _, knowledge = _modules(family)
    spec: RG.LadderSpec = knowledge.SPEC
    drawn = graph.build().compile().get_graph()
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
        assert set(r.tools) <= set(NAMES), f"{family}.{r.name} declares tools the harness lacks: {set(r.tools) - set(NAMES)}"
    space = getattr(knowledge, "SPACE", None)
    if space is not None:
        assert {d.decided_by for d in space.dimensions.values()} <= set(spec.names)
    # the budgets the episodes read are the declared ones, and every rung with tools reads its budget
    source = inspect.getsource(nodes)
    read = {
        str(n.args[0].value)
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_budget" and n.args and isinstance(n.args[0], ast.Constant)
    }
    assert read <= {r.name for r in spec.judgements()}, f"{family}: a budget is read for {read - {r.name for r in spec.judgements()}}, which are not judgements"
    assert {r.name for r in spec.judgements() if r.tools} <= read, f"{family}: judgements with tools that never read their budget"


def test_the_ladder_tables_say_what_the_declarations_say():
    assert EL.stale() == [], "run make ladder"
