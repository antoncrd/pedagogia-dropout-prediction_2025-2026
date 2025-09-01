# run_pipeline.py
import subprocess
from pathlib import Path
import sys
from typing import Sequence

YEAR = 2024 
UNITS = ['B-CPE-100', 'B-CPE-110', 'B-PSU-100', 'B-PSU-200', 'B-CPE-200'] 

def run_cmd(cmd: Sequence[str]) -> None:
    print("→", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    root = Path(__file__).resolve().parent

    # Dossiers
    data_dir     = root / "data"
    data_json    = data_dir / "data_json"
    data_csv     = data_dir / "data_csv"
    data_fin     = data_dir / "data_fin"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_json.mkdir(parents=True, exist_ok=True)
    data_csv.mkdir(parents=True, exist_ok=True)
    data_fin.mkdir(parents=True, exist_ok=True)

    # Scripts
    get_data_py        = root / "data_loading.utils" / "get_data.py"
    all_json_to_csv_py = root / "data_loading.utils" / "all_json_to_csv.py"
    aggregate_csv_py   = root / "data_loading.utils" / "aggregate_csv.py"
    merged_csv_py      = root / "data_loading.utils" / "merged_csv.py"
     # 1) Récupérer les données
    for unit in UNITS:
        run_cmd([
            sys.executable, str(get_data_py),
            "--year", str(YEAR),
            "--unit", unit
        ])

    # 2) JSON -> CSV
    run_cmd([
        sys.executable, str(all_json_to_csv_py),
        "--input", str(data_json),
        "--output", str(data_csv)
    ])

    # 3) Agrégation
    run_cmd([
        sys.executable, str(aggregate_csv_py),
        "--indir", str(data_csv),
        "--outdir", str(data_fin)
    ])

    # 4) Fusion horizontale ordonnée (-> DATA.csv)
    run_cmd([
        sys.executable, str(merged_csv_py),
        "--indir", str(data_fin),
        "--out", str(data_dir / "DATA.csv")
    ])

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
