#!/usr/bin/env python3
"""
Data Loading Pipeline Script

This script orchestrates the data loading process for the dropout prediction project.
It executes a series of Python scripts to fetch data, convert JSON to CSV, aggregate,
and merge the data into a final CSV file.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

# Default constants
DEFAULT_YEAR = 2024
DEFAULT_UNITS = ['B-CPE-100', 'B-CPE-110', 'B-PSU-100', 'B-PSU-200', 'B-CPE-200']
DEFAULT_DATA_DIR = "data"
DEFAULT_UTILS_DIR = "utils"


def run_cmd(cmd: Sequence[str]) -> None:
    """
    Execute a command using subprocess and print it for logging.

    Args:
        cmd (Sequence[str]): The command to execute as a list of strings.

    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    print("→", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main(year: int, units: List[str], data_dir: str, utils_dir: str) -> None:
    """
    Main function to run the data loading pipeline.

    Args:
        year (int): The academic year for data retrieval.
        units (List[str]): List of academic units to process.
        data_dir (str): Path to the data directory.
        utils_dir (str): Path to the utils directory containing scripts.
    """
    # Determine root directory

    # APRÈS
    script_dir = Path(__file__).resolve().parent      # /app/src
    repo_root  = script_dir.parent                    # /app

    def resolve_under(base: Path, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (base / p)

    data_dir_path   = resolve_under(repo_root, data_dir)         # -> /app/data
    utils_dir_path  = resolve_under(script_dir, utils_dir)       # -> /app/src/utils

    get_data_py        = utils_dir_path / "data_loading_utils" / "get_data.py"
    all_json_to_csv_py = utils_dir_path / "data_loading_utils" / "all_json_to_csv.py"
    aggregate_csv_py   = utils_dir_path / "data_loading_utils" / "aggregate_csv.py"
    merged_csv_py      = utils_dir_path / "data_loading_utils" / "merged_csv.py"

    data_json = data_dir_path / "data_json" / str(year)
    data_csv  = data_dir_path / "data_csv"  / str(year)
    data_fin  = data_dir_path / "data_fin"  / str(year)

    # Step 1: Retrieve data for each unit
    print("Step 1: Retrieving data...")
    for unit in units:
        run_cmd([
            sys.executable, str(get_data_py),
            "--year", str(year),
            "--unit", unit
        ])

    # Step 2: Convert JSON to CSV
    print("Step 2: Converting JSON to CSV...")
    run_cmd([
        sys.executable, str(all_json_to_csv_py),
        "--input", str(data_json),
        "--output", str(data_csv)
    ])

    # Step 3: Aggregate CSV files
    print("Step 3: Aggregating CSV files...")
    run_cmd([
        sys.executable, str(aggregate_csv_py),
        "--indir", str(data_csv),
        "--outdir", str(data_fin)
    ])

    # Step 4: Merge CSV files into final DATA.csv
    print("Step 4: Merging CSV files into final output...")
    run_cmd([
        sys.executable, str(merged_csv_py),
        "--year", str(year)
    ])

    print("Data loading pipeline completed successfully!")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the data loading pipeline for dropout prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Academic year for data retrieval."
    )
    parser.add_argument(
        "--units",
        nargs='+',
        default=DEFAULT_UNITS,
        help="List of academic units to process."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to the data directory."
    )
    parser.add_argument(
        "--utils_dir",
        type=str,
        default=DEFAULT_UTILS_DIR,
        help="Path to the utils directory containing scripts."
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        main(
            year=args.year,
            units=args.units,
            data_dir=args.data_dir,
            utils_dir=args.utils_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)
