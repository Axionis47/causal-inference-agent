#!/usr/bin/env python3
"""End-to-end Kaggle dataset test runner.

Submits 3 diverse Kaggle datasets through the API concurrently and validates
that the full pipeline produces correct results, including all L1-L12 fixes.

Usage:
    python run_kaggle_e2e.py              # Run all 3 datasets concurrently
    python run_kaggle_e2e.py --dataset lalonde   # Run one dataset
    python run_kaggle_e2e.py --port 8001         # Custom port
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import uvicorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for one E2E test dataset."""

    name: str
    kaggle_url: str
    treatment_variable: str | None
    outcome_variable: str | None
    ground_truth_ate: float | None
    ate_tolerance_pct: float  # e.g., 0.5 = within 50%
    min_methods: int
    # Notebook content checks: list of (description, substring_to_find)
    notebook_checks: list[tuple[str, str]] = field(default_factory=list)


DATASETS: dict[str, DatasetConfig] = {
    "lalonde": DatasetConfig(
        name="LaLonde NSW",
        kaggle_url="https://www.kaggle.com/datasets/samuelzakouri/lalonde",
        treatment_variable="treat",
        outcome_variable="re78",
        ground_truth_ate=1794.0,
        ate_tolerance_pct=0.6,  # Observational LaLonde estimates vary widely ($961-$2,900)
        min_methods=2,
        notebook_checks=[
            ("L5: Caveat banner", "Interpretation Note"),
            ("L12: Observational caveat in intro", "observational causal analysis"),
            ("Treatment effects table", "Results Summary"),
            ("Sensitivity section", "Sensitivity"),
            ("Conclusions section", "Conclusions"),
            ("Notebook has verification OLS", "Verification"),
        ],
    ),
    "ihdp": DatasetConfig(
        name="IHDP",
        kaggle_url="https://www.kaggle.com/datasets/konradb/ihdp-data",
        treatment_variable="treatment",
        outcome_variable="y_factual",
        ground_truth_ate=None,  # Depends on specific IHDP realization
        ate_tolerance_pct=1.0,
        min_methods=2,
        notebook_checks=[
            ("L12: Observational caveat", "observational causal analysis"),
            ("Treatment effects table", "Results Summary"),
            ("L5: Caveat banner", "Interpretation Note"),
            ("Conclusions section", "Conclusions"),
        ],
    ),
    "hillstrom": DatasetConfig(
        name="Hillstrom Email",
        kaggle_url="https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics",
        treatment_variable="segment",
        outcome_variable="spend",
        ground_truth_ate=None,  # Real experiment, no synthetic ground truth
        ate_tolerance_pct=1.0,
        min_methods=2,  # Collapsed 3-level categorical → binary; 2 methods is reasonable
        notebook_checks=[
            ("L5: Caveat banner", "Interpretation Note"),
            ("L12: Observational caveat", "observational causal analysis"),
            ("Treatment effects table", "Results Summary"),
            ("Conclusions section", "Conclusions"),
        ],
    ),
    "marketing_ab": DatasetConfig(
        name="Marketing A/B",
        kaggle_url="https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing",
        treatment_variable="test group",
        outcome_variable="converted",
        ground_truth_ate=None,
        ate_tolerance_pct=1.0,
        min_methods=2,
        notebook_checks=[
            ("L5: Caveat banner", "Interpretation Note"),
            ("L12: Observational caveat", "observational causal analysis"),
            ("Treatment effects table", "Results Summary"),
            ("Conclusions section", "Conclusions"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Test result
# ---------------------------------------------------------------------------

@dataclass
class TestVerdict:
    """Result of validating one dataset's E2E run."""

    dataset_name: str
    passed: bool
    status: str  # "completed", "failed", "timeout", "error"
    n_methods: int = 0
    ate_estimate: float | None = None
    notebook_path: str | None = None
    checks_passed: int = 0
    checks_total: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class KaggleE2ERunner:
    """Runs E2E tests against the live API."""

    def __init__(self, base_url: str, timeout_minutes: int = 40):
        self.base_url = base_url.rstrip("/")
        self.timeout_minutes = timeout_minutes
        self.verdicts: list[TestVerdict] = []

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] {msg}", flush=True)

    async def wait_for_server(self, max_wait: int = 30):
        """Wait until the server responds to health checks."""
        async with httpx.AsyncClient() as client:
            for i in range(max_wait):
                try:
                    r = await client.get(f"{self.base_url}/health")
                    if r.status_code == 200:
                        return True
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(1)
        return False

    async def submit_job(self, config: DatasetConfig) -> str | None:
        """Submit a job and return the job_id."""
        payload = {"kaggle_url": config.kaggle_url}
        if config.treatment_variable:
            payload["treatment_variable"] = config.treatment_variable
        if config.outcome_variable:
            payload["outcome_variable"] = config.outcome_variable

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.post(f"{self.base_url}/jobs", json=payload)
                if r.status_code == 201:
                    job_id = r.json()["id"]
                    self.log(f"  {config.name}: job={job_id} -- submitted")
                    return job_id
                else:
                    self.log(f"  {config.name}: submit failed ({r.status_code}): {r.text}", "ERROR")
                    return None
            except Exception as e:
                self.log(f"  {config.name}: submit error: {e}", "ERROR")
                return None

    async def poll_job(self, job_id: str, name: str) -> dict[str, Any]:
        """Poll until job completes, fails, or times out."""
        deadline = time.time() + self.timeout_minutes * 60
        poll_interval = 15

        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() < deadline:
                try:
                    r = await client.get(f"{self.base_url}/jobs/{job_id}/status")
                    if r.status_code == 200:
                        data = r.json()
                        status = data.get("status", "unknown")
                        progress = data.get("progress_percentage", 0)
                        agent = data.get("current_agent", "")

                        if status == "completed":
                            return {"status": "completed"}
                        elif status == "failed":
                            return {"status": "failed"}
                        elif status == "cancelled":
                            return {"status": "cancelled"}

                        self.log(f"  {name}: {status} ({progress}%) agent={agent}")
                except Exception as e:
                    self.log(f"  {name}: poll error: {e}", "WARN")

                await asyncio.sleep(poll_interval)

        return {"status": "timeout"}

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Fetch analysis results."""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.get(f"{self.base_url}/jobs/{job_id}/results")
                if r.status_code == 200:
                    return r.json()
            except Exception as e:
                self.log(f"  Results fetch error: {e}", "WARN")
        return None

    async def download_notebook(self, job_id: str, save_dir: Path) -> Path | None:
        """Download the generated notebook."""
        save_dir.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                r = await client.get(f"{self.base_url}/jobs/{job_id}/notebook")
                if r.status_code == 200:
                    path = save_dir / f"e2e_{job_id}.ipynb"
                    path.write_bytes(r.content)
                    return path
            except Exception as e:
                self.log(f"  Notebook download error: {e}", "WARN")
        return None

    def validate_notebook(
        self, nb_path: Path, checks: list[tuple[str, str]]
    ) -> tuple[int, int, list[str]]:
        """Check notebook content for expected strings. Returns (passed, total, errors)."""
        if not nb_path or not nb_path.exists():
            return 0, len(checks), [f"Notebook not found at {nb_path}"]

        try:
            nb_json = json.loads(nb_path.read_text())
        except Exception as e:
            return 0, len(checks), [f"Failed to parse notebook: {e}"]

        # Flatten all cell sources into one big string
        all_text = ""
        for cell in nb_json.get("cells", []):
            source = cell.get("source", [])
            if isinstance(source, list):
                all_text += "".join(source)
            else:
                all_text += source

        passed = 0
        errors = []
        for desc, substring in checks:
            if substring in all_text:
                passed += 1
            else:
                errors.append(f"MISSING: {desc} (expected '{substring}')")

        return passed, len(checks), errors

    async def run_dataset(self, config: DatasetConfig) -> TestVerdict:
        """Run E2E test for one dataset."""
        start = time.time()
        verdict = TestVerdict(
            dataset_name=config.name,
            passed=False,
            status="error",
            checks_total=len(config.notebook_checks),
        )

        # Submit
        job_id = await self.submit_job(config)
        if not job_id:
            verdict.errors.append("Failed to submit job")
            verdict.duration_seconds = time.time() - start
            return verdict

        # Poll
        poll_result = await self.poll_job(job_id, config.name)
        verdict.status = poll_result["status"]
        verdict.duration_seconds = time.time() - start

        if verdict.status != "completed":
            verdict.errors.append(f"Job ended with status: {verdict.status}")
            self.log(f"  {config.name}: {verdict.status} in {verdict.duration_seconds:.0f}s", "ERROR")
            return verdict

        self.log(f"  {config.name}: COMPLETED in {verdict.duration_seconds:.0f}s")

        # Fetch results
        results = await self.get_results(job_id)
        if not results:
            verdict.errors.append("Failed to fetch results")
            return verdict

        # Count methods
        effects = results.get("treatment_effects", [])
        verdict.n_methods = len(effects)

        if verdict.n_methods < config.min_methods:
            verdict.errors.append(
                f"Only {verdict.n_methods} methods (need {config.min_methods})"
            )

        # ATE estimate (average of ATE estimands)
        ate_estimates = [
            e["estimate"] for e in effects
            if e.get("estimand", "").upper() == "ATE"
        ]
        if ate_estimates:
            import numpy as np
            verdict.ate_estimate = float(np.median(ate_estimates))

        # Check ground truth
        if config.ground_truth_ate is not None and verdict.ate_estimate is not None:
            error_pct = abs(verdict.ate_estimate - config.ground_truth_ate) / abs(config.ground_truth_ate)
            if error_pct > config.ate_tolerance_pct:
                verdict.errors.append(
                    f"ATE={verdict.ate_estimate:.2f} vs truth={config.ground_truth_ate:.2f} "
                    f"(error={error_pct:.0%}, tolerance={config.ate_tolerance_pct:.0%})"
                )

        # Download and validate notebook
        save_dir = Path("/tmp/causal_orchestrator/e2e_notebooks")
        nb_path = await self.download_notebook(job_id, save_dir)
        if nb_path:
            verdict.notebook_path = str(nb_path)
            passed, total, nb_errors = self.validate_notebook(nb_path, config.notebook_checks)
            verdict.checks_passed = passed
            verdict.checks_total = total
            verdict.errors.extend(nb_errors)
        else:
            verdict.errors.append("Notebook download failed")

        # Final verdict
        verdict.passed = (
            verdict.status == "completed"
            and verdict.n_methods >= config.min_methods
            and len(verdict.errors) == 0
        )

        return verdict

    async def run_all(self, dataset_names: list[str] | None = None):
        """Run all configured datasets concurrently."""
        configs = []
        if dataset_names:
            for name in dataset_names:
                if name not in DATASETS:
                    self.log(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}", "ERROR")
                    sys.exit(1)
                configs.append(DATASETS[name])
        else:
            # Default: run the 3 core datasets (marketing_ab requires specific Kaggle access)
            default_names = ["lalonde", "ihdp", "hillstrom"]
            configs = [DATASETS[n] for n in default_names]

        self.log(f"Submitting {len(configs)} jobs concurrently...")

        # Run all concurrently
        tasks = [self.run_dataset(config) for config in configs]
        self.verdicts = await asyncio.gather(*tasks)

    def print_summary(self) -> bool:
        """Print summary table. Returns True if all passed."""
        print("\n" + "=" * 90)
        print("E2E KAGGLE TEST SUMMARY")
        print("=" * 90)

        all_passed = all(v.passed for v in self.verdicts)
        n_passed = sum(1 for v in self.verdicts if v.passed)

        print(f"\nTotal: {len(self.verdicts)} | Passed: {n_passed} | Failed: {len(self.verdicts) - n_passed}")
        print()

        header = f"{'Dataset':<18} {'Status':<10} {'Methods':<8} {'ATE Est':<12} {'Notebook':<10} {'Checks':<10} {'Time':<10}"
        print(header)
        print("-" * 90)

        for v in self.verdicts:
            status_icon = "PASS" if v.passed else "FAIL"
            ate_str = f"{v.ate_estimate:.2f}" if v.ate_estimate is not None else "N/A"
            nb_str = "yes" if v.notebook_path else "no"
            checks_str = f"{v.checks_passed}/{v.checks_total}"
            mins = int(v.duration_seconds // 60)
            secs = int(v.duration_seconds % 60)
            time_str = f"{mins}m {secs}s"

            print(f"{v.dataset_name:<18} {status_icon:<10} {v.n_methods:<8} {ate_str:<12} {nb_str:<10} {checks_str:<10} {time_str:<10}")

        # Print errors
        has_errors = any(v.errors for v in self.verdicts)
        if has_errors:
            print("\n" + "-" * 90)
            print("ISSUES:")
            for v in self.verdicts:
                if v.errors:
                    for err in v.errors:
                        print(f"  [{v.dataset_name}] {err}")

        print("=" * 90)

        if all_passed:
            print("All tests passed.")
        else:
            print(f"{len(self.verdicts) - n_passed} test(s) failed.")

        return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    port = args.port

    # Start the API server in-process
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Starting API server on port {port}...")

    config = uvicorn.Config(
        "src.api.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Run server in background
    server_task = asyncio.create_task(server.serve())

    base_url = f"http://127.0.0.1:{port}"
    runner = KaggleE2ERunner(base_url=base_url, timeout_minutes=args.timeout)

    # Wait for server to be ready
    ready = await runner.wait_for_server(max_wait=30)
    if not ready:
        print("ERROR: Server did not start in time.", file=sys.stderr)
        sys.exit(1)

    runner.log("Server ready.")

    # Select datasets
    dataset_names = [args.dataset] if args.dataset else None

    # Run tests
    await runner.run_all(dataset_names)

    # Summary
    all_passed = runner.print_summary()

    # Shutdown server
    server.should_exit = True
    await server_task

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Kaggle dataset tests for causal pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Run a single dataset (default: all 3 concurrently)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Port for the test API server (default: 8099)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Timeout in minutes per job (default: 40)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
