"""Real exploration acceptance through the owned main-chat HTTP boundary.

uv run python -m causal_agent.evals.explore <conversation-uuid> "Plot age against education"
"""

import argparse
import json
import time

import httpx


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("conversation")
    parser.add_argument("question")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    with httpx.Client(base_url=args.url, timeout=60) as client:
        path = f"/api/conversations/{args.conversation}"
        response = client.post(path + "/messages", json={"text": args.question})
        response.raise_for_status()
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            response = client.get(path)
            response.raise_for_status()
            view = response.json()
            if view["stage"] != "busy":
                print(json.dumps(view, indent=2))
                if view["stage"] == "error":
                    raise SystemExit(1)
                return
            time.sleep(1)
    raise TimeoutError("The conversation did not finish within fifteen minutes")


if __name__ == "__main__":
    main()
