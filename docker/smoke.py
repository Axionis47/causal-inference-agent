"""Build and exercise the current checkout in a disposable Compose project without model calls.

This creates and removes only its own randomly named containers and volumes.
Run from the repository: uv run python docker/smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CSV = b"id,value\n1,10\n2,20\n"


def main() -> None:
    project = "causal-smoke-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="causal-compose-") as scratch:
        override = Path(scratch) / "compose.yaml"
        override.write_text(
            'services:\n  app:\n    ports: !override ["127.0.0.1::8000"]\n    environment:\n      LANGSMITH_TRACING: "false"\n      LANGSMITH_API_KEY: ""\n'
        )
        compose = ["docker", "compose", "--env-file", os.devnull, "-p", project, "-f", str(ROOT / "compose.yaml"), "-f", str(override)]

        clean_env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "TMPDIR", "SYSTEMROOT"}}

        def command(*args: str, check: bool = True) -> str:
            result = subprocess.run([*compose, *args], env=clean_env, capture_output=True, text=True, timeout=900)
            if check and result.returncode:
                raise RuntimeError(f"Docker command failed: {args}\n{result.stdout}\n{result.stderr}")
            return result.stdout.strip()

        def base_url() -> str:
            return "http://" + command("port", "app", "8000")

        def browser_evidence(journey: str):
            evidence = os.environ.get("PHASE4_EVIDENCE_DIR")
            if not evidence:
                return {}, []
            folder = Path(evidence).resolve() / journey
            folder.mkdir(parents=True, exist_ok=True)
            return {"PLAYWRIGHT_JUNIT_OUTPUT_FILE": str(folder / "results.xml")}, ["--reporter=list,junit", f"--output={folder / 'traces'}"]

        def request(path: str, body: bytes | None = None, headers: dict | None = None):
            req = Request(base_url() + path, data=body, headers=headers or {})
            with urlopen(req, timeout=30) as response:
                return response.headers.get("Content-Type", ""), response.read()

        try:
            print(f"Starting disposable project {project}", flush=True)
            command("build", "app", "sandbox")
            command("up", "-d", "--no-build", "--wait", "--wait-timeout", "180", "app")
            before_ids = {service: command("ps", "-q", service) for service in ("app", "sandbox", "postgres", "objects")}
            assert all(before_ids.values())
            image = subprocess.check_output(["docker", "inspect", "--format", "{{.Image}}", before_ids["app"]], env=clean_env, text=True).strip()
            print(f"Built image: {image}", flush=True)
            browser_env, browser_args = browser_evidence("ingestion")
            subprocess.run(
                ["npm", "--prefix", "web", "exec", "--", "playwright", "test", "ingestion.spec.ts", "--config", "web/playwright.config.ts", *browser_args],
                cwd=ROOT,
                env={**clean_env, **browser_env, "INGESTION_BROWSER_URL": base_url()},
                check=True,
                timeout=120,
            )
            assert json.loads(request("/api/health")[1]) == {"ok": True}
            print(command("exec", "-T", "app", "python", "/usr/local/bin/causal-sandbox-probe.py"), flush=True)
            assert "text/html" in request("/")[0]
            assert len(json.loads(request("/api/datasets")[1])["items"]) == 2
            boundary = uuid.uuid4().hex
            fields = {"title": "Container smoke", "context": json.dumps(" Original account\nunchanged "), "idempotency_key": uuid.uuid4().hex}
            body = b""
            for name, value in fields.items():
                body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            body += (
                (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()
                + CSV
                + f"\r\n--{boundary}--\r\n".encode()
            )
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
            saved = json.loads(request("/api/datasets", body, headers)[1])
            assert saved["status"] == "saved" and saved["initial_context"] == json.loads(fields["context"])
            assert json.loads(request("/api/datasets", body, headers)[1]) == saved
            path = "/api/datasets/" + saved["dataset_id"]
            assert request(path + "/file")[1] == CSV
            assert saved["dataset_id"] in {item["dataset_id"] for item in json.loads(request("/api/datasets")[1])["items"]}
            profile = json.loads(request(path + "/profile", b"")[1])
            assert profile["status"] == "complete" and profile["profile"]["dataset"]["rows"] == 2
            column_path = path + "/profiles/" + profile["profile"]["profile_id"] + "/columns/" + profile["profile"]["columns"][1]["column_id"]
            column_profile = json.loads(request(column_path)[1])
            assert column_profile["result"]["numeric"]["mean"] == 15
            dataset_ids = [item["dataset_id"] for item in json.loads(request("/api/datasets")[1])["items"]]
            print(command("exec", "-T", "app", "python", "-c", (ROOT / "docker/context_fixture.py").read_text(), *dataset_ids), flush=True)
            context = json.loads(request(path + "/context")[1])
            context_path = path + "/contexts/" + context["pack"]["revision_id"]
            context_detail = json.loads(request(context_path)[1])
            context_column_path = context_path + "/columns/" + context["pack"]["columns"][0]["column_id"]
            context_column = json.loads(request(context_column_path)[1])
            browser_env, browser_args = browser_evidence("context")
            subprocess.run(
                ["npm", "--prefix", "web", "exec", "--", "playwright", "test", "context.spec.ts", "--config", "web/playwright.config.ts", *browser_args],
                cwd=ROOT,
                env={**clean_env, **browser_env, "INGESTION_BROWSER_URL": base_url()},
                check=True,
                timeout=120,
            )
            browser_env, browser_args = browser_evidence("chat")
            subprocess.run(
                ["npm", "--prefix", "web", "exec", "--", "playwright", "test", "chat.spec.ts", "--config", "web/playwright.config.ts", *browser_args],
                cwd=ROOT, env={**clean_env, **browser_env, "INGESTION_BROWSER_URL": base_url()}, check=True, timeout=120,
            )
            chats = [chat for dataset_id in dataset_ids for chat in json.loads(request(f"/api/datasets/{dataset_id}/conversations")[1])]
            chat_views = {c["id"]: json.loads(request(f"/api/conversations/{c['id']}")[1]) for c in chats}
            chat_messages = {c["id"]: json.loads(request(f"/api/conversations/{c['id']}/messages")[1]) for c in chats}
            assert chat_views, "Chat browser journey must create a reviewed conversation"
            print("Recreating application, sandbox, database and object store with their existing volumes", flush=True)
            command("up", "-d", "--no-build", "--force-recreate", "--wait", "--wait-timeout", "180", "postgres", "objects", "sandbox", "app")
            after_ids = {service: command("ps", "-q", service) for service in before_ids}
            assert all(after_ids[service] != before_ids[service] for service in before_ids), "Every service must be replaced"
            assert saved["dataset_id"] in {item["dataset_id"] for item in json.loads(request("/api/datasets")[1])["items"]}
            assert json.loads(request(path)[1]) == saved
            assert json.loads(request(path + "/profile")[1]) == profile
            assert json.loads(request(column_path)[1]) == column_profile
            assert json.loads(request(path + "/context")[1]) == context
            assert json.loads(request(context_path)[1]) == context_detail
            assert json.loads(request(context_column_path)[1]) == context_column
            assert request(path + "/file")[1] == CSV
            for chat_id in chat_views:
                assert json.loads(request(f"/api/conversations/{chat_id}")[1]) == chat_views[chat_id]
                assert json.loads(request(f"/api/conversations/{chat_id}/messages")[1]) == chat_messages[chat_id]
            assert json.loads(request("/api/datasets", body, headers)[1]) == saved
            command("exec", "-T", "app", "python", "-m", "causal_agent.server.ingestion", "cleanup")
            assert request(path + "/file")[1] == CSV
            print(
                "PASS: ingestion, immutable originals, owned dataset/column profiles and context, and persistence across all service replacements. No model calls.",
                flush=True,
            )
        except Exception:
            print(command("logs", "--tail", "60", "app", check=False), flush=True)
            raise
        finally:
            print(f"Removing only disposable project {project} and its volumes", flush=True)
            command("down", "--volumes", "--remove-orphans")


if __name__ == "__main__":
    main()
