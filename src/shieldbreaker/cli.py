"""Executable paper-aligned analysis and synthetic validation."""

import argparse
import json
from shieldbreaker.config import AnalysisConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="shieldbreaker")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "analyze",
        help="Run baseline calibration, localization, semantics import, and recurrence",
    )
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model", choices=["ngram", "gpt2", "lstm"], default="gpt2")
    run.add_argument("--model-path")
    run.add_argument(
        "--inventory", help="Optional external record-level candidate inventory"
    )
    run.add_argument("--device")
    export = sub.add_parser(
        "semantic-requests",
        help="Export fixed GPT-5.4 requests; makes no external API calls",
    )
    export.add_argument("--manifest", required=True)
    export.add_argument("--output-dir", required=True)
    sim = sub.add_parser(
        "simulate",
        help="Run abstract synthetic experiments, not commercial EDR executions",
    )
    sim.add_argument(
        "study",
        choices=[
            "component",
            "connected",
            "discrimination",
            "confirmation",
            "adaptation",
            "all",
        ],
    )
    sim.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        from shieldbreaker.pipeline import run_analysis

        print(
            json.dumps(
                run_analysis(
                    args.manifest,
                    args.output_dir,
                    args.model,
                    args.model_path,
                    AnalysisConfig(),
                    args.device,
                    args.inventory,
                ),
                indent=2,
            )
        )
    elif args.command == "semantic-requests":
        from shieldbreaker.pipeline import export_semantic_requests

        export_semantic_requests(args.manifest, args.output_dir)
    else:
        from shieldbreaker.experiments.run import run_studies

        run_studies(args.study, args.output_dir)


if __name__ == "__main__":
    main()
