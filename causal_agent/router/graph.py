"""Moved: the routing lives in causal_agent.desk.route. This module stays so `langgraph.json`, `router/run.py`, and the chat's
pipeline keep working until the desk graph replaces them (stage 4 of docs/desk-redesign.md)."""

from causal_agent.desk.route import build, compile_local, graph  # noqa: F401
