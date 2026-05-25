import argparse
import json
from pathlib import Path


DEFAULT_EDR_PROCESSES = [
    "NisSrv.exe",
    "MsMpEng.exe",
    "SecurityHealthService.exe",
    "SecurityHealthSystray.exe",
]


def extract_process_name(process_path: str) -> str:
    if not process_path:
        return ""
    process_path = process_path.replace("/", "\\")
    return process_path.split("\\")[-1]


def extract_edr_processes(input_file, output_file, target_processes, append=False):
    """
    Extract records related to EDR processes from a JSONL input file.
    """
    extracted_count = 0
    total_count = 0

    mode = 'a' if append else 'w'

    print(f"Start processing file: {input_file}")
    print(f"Write mode: {'append' if append else 'overwrite'}")
    print(f"Target EDR processes: {', '.join(target_processes)}")
    print("-" * 60)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, mode, encoding='utf-8') as outfile:

        for line in infile:
            total_count += 1

            if total_count % 10000 == 0:
                print(f"Processed {total_count} records, extracted {extracted_count} records...")

            try:
                data = json.loads(line.strip())

                process_path = data.get('ProcessName', '')
                process_name = extract_process_name(process_path)

                plugin = data.get('Plugin', '')
                if plugin == 'sysret':
                    continue

                if process_name in target_processes:
                    outfile.write(line)
                    extracted_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {total_count}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {total_count}: {e}")

    print("-" * 60)
    print(f"Finished processing file: {input_file}")
    print(f"Total records processed: {total_count}")
    print(f"EDR records extracted: {extracted_count}\n")
    return extracted_count, total_count


def iter_input_files(input_dir: str, pattern: str, recursive: bool):
    base_dir = Path(input_dir)
    if recursive:
        files = sorted(p for p in base_dir.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in base_dir.glob(pattern) if p.is_file())

    return [p for p in files if not p.name.startswith("._")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EDR process records from JSONL-style logs")
    parser.add_argument("--input_dir", type=str, default="datasets/baseline", help="Input directory")
    parser.add_argument("--output_file", type=str, default="datasets/baseline/edr_only.jsonl", help="Output JSONL file")
    parser.add_argument("--pattern", type=str, default="*.jsonl", help="Input filename pattern, e.g. *.jsonl or drakvuf.json")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input_dir")
    parser.add_argument("--processes", nargs="+", default=DEFAULT_EDR_PROCESSES, help="Target EDR process names")
    args = parser.parse_args()

    input_files = iter_input_files(args.input_dir, args.pattern, args.recursive)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_files:
        print(f"Error: no input files matched {args.pattern} under {Path(args.input_dir).resolve()}")
        raise SystemExit(1)

    print(f"Found {len(input_files)} files")

    total_records = 0
    total_extracted = 0
    first_file = True

    for input_path in input_files:
        extracted_count, record_count = extract_edr_processes(
            input_file=str(input_path),
            output_file=str(output_path),
            target_processes=args.processes,
            append=not first_file,
        )
        first_file = False
        total_records += record_count
        total_extracted += extracted_count

    print("-" * 60)
    print(f"[SUCCESS] EDR process data has been successfully extracted to: {output_path.resolve()}")
    print(f"[SUMMARY] Files={len(input_files)} Records={total_records} Extracted={total_extracted}")
