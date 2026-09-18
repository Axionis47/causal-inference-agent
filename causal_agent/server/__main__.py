"""Serve the desk: `uv run python -m causal_agent.server [--port 8000] [--reload]` (the port defaults to $PORT). Build the page first with
`npm run build --prefix web`, or run the Vite dev server beside this for hot reload."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")), help="defaults to $PORT, then 8000")
    ap.add_argument("--reload", action="store_true", help="restart on code changes under causal_agent/")
    args = ap.parse_args(argv)
    if args.reload:
        uvicorn.run("causal_agent.server.app:create_app", factory=True, host=args.host, port=args.port, reload=True, reload_dirs=["causal_agent"])
    else:
        from causal_agent.server.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
