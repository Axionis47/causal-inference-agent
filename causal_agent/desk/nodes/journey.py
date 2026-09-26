"""The conversation, node by node, in the order the desk walks them. The nodes live in question.py, interview.py and run.py;
this module is where the graph and the after-run nodes read them."""

from __future__ import annotations

from causal_agent.desk.nodes.interview import ask, check, convince, explain, infer, listen, probe_fit
from causal_agent.desk.nodes.question import ask_question, load, read_question
from causal_agent.desk.nodes.run import after_run, ask_back, gate, handoff, run
from causal_agent.desk.nodes.shared import CAT, QUIT_WORDS, kinds_text

__all__ = [
    "CAT",
    "QUIT_WORDS",
    "after_run",
    "ask",
    "ask_back",
    "ask_question",
    "check",
    "convince",
    "explain",
    "gate",
    "handoff",
    "infer",
    "kinds_text",
    "listen",
    "load",
    "probe_fit",
    "read_question",
    "run",
]
