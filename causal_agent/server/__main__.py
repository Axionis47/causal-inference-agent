"""Serve the desk: `uv run python -m causal_agent.server [--port 8000] [--reload]` (the port defaults to $PORT). Build the page first with
`npm run build --prefix web`, or run the Vite dev server beside this for hot reload."""

from __future__ import annotations

import argparse

import uvicorn

from causal_agent.common import config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=config.get().port, help="defaults to $PORT, then 8000")
    ap.add_argument("--reload", action="store_true", help="restart on code changes under causal_agent/")
    ap.add_argument("--openapi", action="store_true", help="print the OpenAPI schema as JSON and exit; the web's types are generated from it")
    args = ap.parse_args(argv)
    if args.openapi:
        import json

        from causal_agent.server.app import create_app

        print(json.dumps(create_app().openapi(), indent=2))
        return
    if args.reload:
        uvicorn.run("causal_agent.server.app:create_app", factory=True, host=args.host, port=args.port, reload=True, reload_dirs=["causal_agent"])
    else:
        from causal_agent.server.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
