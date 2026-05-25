#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

INPUT_FILE = r"D:\matengfei\LeakEDR\LeakEDR\123456789.json"
OUTPUT_FILE = r"D:\matengfei\LeakEDR\LeakEDR\output-123456789.json"

TARGET_PLUGIN = "syscall"
TARGET_PROCESS = r"\Device\HarddiskVolume2\Windows\System32\notepad.exe"


def match_record(obj):
    return (
        isinstance(obj, dict)
        and obj.get("Plugin") == TARGET_PLUGIN
        and obj.get("ProcessName") == TARGET_PROCESS
    )


def process_json_lines(input_path, output_path):
    count = 0
    total = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if match_record(obj):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    return count, total, skipped


def process_json_array(input_path, output_path):
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    if not isinstance(data, list):
        raise ValueError("文件不是 JSON 数组格式")

    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in data:
            if match_record(obj):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1

    return count, len(data), 0


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return

    # 先尝试按 JSON Lines 处理；如果失败，再尝试按 JSON 数组处理
    try:
        count, total, skipped = process_json_lines(input_path, output_path)
        print("按 JSON Lines 格式处理完成")
    except Exception:
        count, total, skipped = process_json_array(input_path, output_path)
        print("按 JSON 数组格式处理完成")

    print(f"输出文件: {output_path}")
    print(f"匹配行数: {count}")
    print(f"总读取记录数: {total}")
    print(f"跳过无法解析的行数: {skipped}")


if __name__ == "__main__":
    main()
