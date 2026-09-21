"""The evals: one runner for every family. A family keeps its cases, its stored hand-offs, its summariser and the evaluators
that are its own; what is shared is here. Outside the layers: this package may import anything.

    uv run python -m causal_agent.evals.dataset <family>            # upload the family's cases to LangSmith, idempotent
    uv run python -m causal_agent.evals.run <family>                # run and score them
    uv run python -m causal_agent.evals.lane <family> <dataset> "<question>"   # one run from the command line
    uv run python -m causal_agent.evals.lane <family> --handoff handoff.json   # the lane alone, from a stored hand-off
"""

from dotenv import load_dotenv

load_dotenv()
