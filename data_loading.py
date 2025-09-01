# run_pipeline.py
import subprocess

def run_cmd(cmd: str):
    print(f"→ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Récupérer les données
    run_cmd("python get_data.py")

    # 2. Transformer les JSON en CSV
    run_cmd("python all_json_to_csv.py data/data_json/ data/data_csv/")

    # 3. Agréger les CSV par module
    run_cmd("python aggregate_csv.py --indir data/data_csv/ --outdir data/")

if __name__ == "__main__":
    main()
