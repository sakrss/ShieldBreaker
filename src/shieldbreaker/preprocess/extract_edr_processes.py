"""Filter baseline API records while preserving execution boundaries and metadata."""

import argparse
from pathlib import Path
from shieldbreaker.traces import partition_records, read_records, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--processes", nargs="+", required=True)
    args = parser.parse_args()
    base = Path(args.input_dir)
    output = Path(args.output_file).resolve()
    files = sorted(
        base.rglob(args.pattern) if args.recursive else base.glob(args.pattern)
    )
    files = [
        p
        for p in files
        if p.is_file() and p.resolve() != output and not p.name.startswith("._")
    ]
    if not files:
        raise ValueError("No input executions found")

    def extracted():
        for path in files:
            edr, _ = partition_records(read_records(path), args.processes)
            for event in edr:
                yield {
                    **event.metadata,
                    "ExecutionID": path.relative_to(base).as_posix(),
                    "Role": "edr",
                }

    write_jsonl(output, extracted())


if __name__ == "__main__":
    main()
