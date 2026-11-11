"""Evaluate a frozen candidate inventory on held-out execution observations."""

import argparse
import json
from pathlib import Path
from shieldbreaker.analysis.recurrence import evaluate_held_out
from shieldbreaker.traces import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    candidates = json.loads(Path(args.candidates).read_text())
    observations = [
        json.loads(line)
        for line in Path(args.observations).read_text().splitlines()
        if line.strip()
    ]
    if any(row["split"] != "test" for row in observations):
        raise ValueError("Held-out evaluation accepts test observations only")
    write_json(args.output_file, evaluate_held_out(candidates, observations))


if __name__ == "__main__":
    main()
