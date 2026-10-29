"""Check HTTP liveness and the built page without calling a model."""

import json
from urllib.request import urlopen


def main() -> None:
    with urlopen("http://127.0.0.1:8000/api/health", timeout=2) as response:
        if json.load(response) != {"ok": True}:
            raise SystemExit("API health check failed")
    with urlopen("http://127.0.0.1:8000/api/ready", timeout=4) as response:
        if not json.load(response).get("ok"):
            raise SystemExit("Exploration runtime unavailable")
    with urlopen("http://127.0.0.1:8000/", timeout=2) as response:
        if response.status != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            raise SystemExit("Built frontend is unavailable")


if __name__ == "__main__":
    main()
