"""Cross-cutting infrastructure used by every agent and the orchestrator.

Substrate concerns are deliberately separated from agent logic so that
trace, budget, tool-log, retry, events, and ctx can be unit-tested in
isolation and so that the orchestrator can reason about them without
pulling in agent code.
"""
