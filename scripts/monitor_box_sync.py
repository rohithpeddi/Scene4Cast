#!/usr/bin/env python3
"""
Monitor script for Box dataset synchronization processes.

Features:
  - Inspects background sync agent log files and process statuses.
  - Queries Box API for current live state in folder 380446756239.
  - Displays structured progress tables every 60 seconds (when --loop is set) or once.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.box_sync_common import get_box_client, human_size
from scripts.sync_all_datasets_box import (
    LOG_DIR,
    TARGET_ROOT_FOLDER_ID,
    check_status,
)


def print_banner(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}  -  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def report_sync_progress(client=None):
    if client is None:
        client = get_box_client()

    print_banner("Box Sync Monitoring Report")

    # 1. Check agent logs
    print("\n--- Agent Execution Logs ---")
    for item in [1, 2, 3, 4]:
        log_file = LOG_DIR / f"agent_item_{item}.log"
        if not log_file.exists():
            print(f"Agent {item}: No log file yet ({log_file})")
            continue

        size = log_file.stat().st_size
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%H:%M:%S")
        lines = log_file.read_text().strip().split("\n")
        last_line = lines[-1] if lines else "Empty"
        status = "COMPLETED" if any("Finished" in l or "Successfully" in l for l in lines[-3:]) else "RUNNING / IN PROGRESS"
        if any("Error" in l or "Exception" in l for l in lines[-5:]):
            status = "ERROR / WARNING"

        print(f"Agent {item} [{status}] (Log: {log_file.name}, modified: {mtime}, size: {size}B):")
        print(f"   Latest: {last_line[:90]}")

    # 2. Check Box folder state
    print("\n--- Box Folder 380446756239 State ---")
    try:
        check_status(client)
    except Exception as e:
        print(f"Warning querying Box: {e}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Box dataset sync progress")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in a loop reporting every --interval seconds",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum loop iterations (default: unlimited)",
    )
    args = parser.parse_args()

    client = get_box_client()

    if not args.loop:
        report_sync_progress(client)
        return

    iteration = 0
    try:
        while True:
            iteration += 1
            report_sync_progress(client)
            if args.max_iterations and iteration >= args.max_iterations:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")


if __name__ == "__main__":
    main()
