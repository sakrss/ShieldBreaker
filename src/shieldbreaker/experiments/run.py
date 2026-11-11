"""Run paper-specified synthetic designs and persist full new ledgers."""

from pathlib import Path
import platform
import numpy
from shieldbreaker.traces import write_json
from shieldbreaker.experiments import component, connected, discrimination, adaptation


def run_studies(study, output_dir):
    studies = {
        "component": component.run,
        "connected": connected.run,
        "discrimination": discrimination.run,
        "confirmation": discrimination.run_confirmation,
        "adaptation": adaptation.run,
    }
    selected = studies if study == "all" else {study: studies[study]}
    for name, run in selected.items():
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        write_json(
            path / "environment.json",
            {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "source": "new synthetic implementation following stated design; original program unavailable",
            },
        )
        print(f"Running {name} -> {path}", flush=True)
        run(path)
        print(f"Completed {name}", flush=True)
