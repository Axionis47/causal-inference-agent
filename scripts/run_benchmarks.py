#!/usr/bin/env python3
"""Script to run benchmark evaluation suite.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --dataset ihdp
    python scripts/run_benchmarks.py --datasets all
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))


async def main():
    parser = argparse.ArgumentParser(description="Run causal inference benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to benchmark (e.g., ihdp, lalonde)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of datasets or 'all'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    from benchmarks.datasets.loader import BenchmarkDatasetLoader
    from benchmarks.runners.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner()
    loader = BenchmarkDatasetLoader()

    if args.dataset:
        # Run single dataset
        dataset_map = {
            "ihdp": loader.load_ihdp,
            "lalonde": loader.load_lalonde,
            "twins": loader.load_twins,
            "card_iv": loader.load_card_iv,
            "card_krueger_did": loader.load_card_krueger_did,
            "acic_2016": loader.load_acic_2016,
            "time_series_climate": loader.load_time_series_climate,
            "news_continuous": loader.load_news_continuous,
        }

        if args.dataset not in dataset_map:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(dataset_map.keys())}")
            sys.exit(1)

        dataset = dataset_map[args.dataset]()
        print(f"Running benchmark on: {dataset.name}")
        result = await runner.run_single(dataset)

        print(f"\nResult: {'PASS' if result.success else 'FAIL'}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Estimated effects: {len(result.estimated_effects)}")
            for effect in result.estimated_effects:
                print(f"  - {effect['method']}: {effect['estimate']:.4f}")
    else:
        # Run all benchmarks
        print("Running all 8 benchmark datasets...")
        report = await runner.run_all()

        # Save report
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        runner.save_report(report, str(output_file))

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Datasets tested: {report.datasets_tested}")
        print(f"Datasets passed: {report.datasets_passed}")
        print(f"Success rate: {report.overall_metrics.get('success_rate', 0):.1%}")

        if report.overall_metrics.get("avg_abs_bias") is not None:
            print(f"Average absolute bias: {report.overall_metrics['avg_abs_bias']:.4f}")

        if report.overall_metrics.get("avg_coverage") is not None:
            print(f"Average CI coverage: {report.overall_metrics['avg_coverage']:.1%}")

        print("\nPer-dataset results:")
        for result in report.results:
            status = "PASS" if result.success else "FAIL"
            print(f"  {result.dataset_name}: {status}")
            if result.error and args.verbose:
                print(f"    Error: {result.error}")

        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
