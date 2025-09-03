#!/usr/bin/env python3
"""
Model Production Main Script

This script performs the main model production pipeline for dropout prediction.
It includes data loading, preprocessing, clustering, and model training using various techniques.

Usage:
    python model_production_main_julien.py --year 24 --n_clusters 4 --min_cluster_size 50 --data_file data/DATA.csv

Author: Julien (based on original model_production_main.py)
Date: September 2025
"""

import argparse
import re
import numpy as np
import pandas as pd
import json
from pathlib import Path

from tqdm import tqdm


# Import functions from models_production_utils (assuming they are defined there)
# If not, they need to be defined or imported properly
from utils.models_production_utils import (
    run_analysis_w,
    OneSidedSPCI_LGBM_Offline,
    TwoSidedSPCI_RFQuant_Offline,
    train_combined_models
)

from utils.model_production_data_processing_utils import(
    cluster_with_min_size,
    build_umap_windows_by_suffix
)

# Default constants
DEFAULT_YEAR = 24
DEFAULT_N_CLUSTERS = 4
DEFAULT_MIN_CLUSTER_SIZE = 50
DEFAULT_DATA_FILE = "data/DATA.csv"
DEFAULT_THRESHOLD = 0.5  # Assuming a default threshold; adjust as needed 


def load_and_preprocess_data(data_file: str, year: int) -> pd.DataFrame:
    """
    Load and preprocess the data from CSV file.

    Args:
        data_file (str): Path to the CSV data file.
        year (int): Academic year for filtering.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df_ = pd.read_csv(data_file)
    df = df_.copy()

    # Normalize mark columns
    mark_cols = [c for c in df.columns if c.endswith("mark")]
    df[mark_cols] = df[mark_cols].div(df[mark_cols].mean())

    # Handle NaN values based on year
    nb_nan_par_ligne = df.isna().sum(axis=1)
    print("Maximum number of NaN in a row:", max(nb_nan_par_ligne))
    if year == 24:
        df = df[nb_nan_par_ligne < 495]
    elif year == 23:
        df = df[nb_nan_par_ligne < 130]

    df["source"] = "real"
    print("Data preprocessing done")
    return df


def prepare_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Prepare feature matrix X based on the year.

    Args:
        df (pd.DataFrame): Input DataFrame.
        year (int): Academic year.

    Returns:
        pd.DataFrame: Feature matrix X.
    """
    if year == 24:
        # CPE 2024
        dfcpool = df[[c for c in df.columns if c.startswith("B-CPE-100")]]
        pat = re.compile(r"B-CPE-100_cpoolday\d+_\d{2} - task\d+_passed")
        cols_keep = [c for c in dfcpool.columns if not pat.match(c)]
        dfcpool_mark = dfcpool[cols_keep]
        X = dfcpool_mark.fillna(0)
    elif year == 23:
        # CPE 2023
        X = df[[c for c in df.columns if c.startswith("B-CPE-110_settingup")]].fillna(0)
    else:
        raise ValueError(f"Unsupported year: {year}")
    return X


def main(
    year: int,
    n_clusters: int,
    min_cluster_size: int,
    data_file: str,
    threshold: float,
    w1: int = 3,
    w2: int = 10,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
) -> None:
    """
    Main function to run the model production pipeline.

    Args:
        year (int): Academic year.
        n_clusters (int): Number of clusters.
        min_cluster_size (int): Minimum cluster size.
        data_file (str): Path to data file.
        threshold (float): Threshold for analysis.
        w1 (int): Window size for first analysis.
        w2 (int): Window size for second analysis.
        alpha1 (float): Alpha for first SPCI model.
        alpha2 (float): Alpha for second SPCI model.
    """
    # Load and preprocess data
    df1 = load_and_preprocess_data(data_file, year)

    # Prepare features
    X = prepare_features(df1, year)

    # Perform clustering
    df2, info = cluster_with_min_size(
        df1, X, n_clusters=n_clusters, min_cluster_size=min_cluster_size, random_state=42
    )
    print("Reassignment of small clusters done")

    # Run analysis with clustering
    df_detail, df_agg, y_cible2, models_c = run_analysis_w(
        df=df2, threshold=threshold, do_plot=False
    )
    models = {}
    models["+ clustering"] = models_c
    print("Models with plain + clustering done")

    # Build UMAP windows and SPCI next grade
    Xt, keys, X_arr, y_arr = build_umap_windows_by_suffix(
        df1, w=w1, H=0, target_col_idx=3, verbose=True
    )

    U_t = []
    for i in tqdm(range(1, len(X_arr) - 1), desc="Processing windows"):
        X_train = np.vstack(X_arr[:i])
        y_train = np.concatenate(y_arr[:i])
        model = OneSidedSPCI_LGBM_Offline(alpha=alpha1, w=300, random_state=42)
        model.fit(X_train, y_train)
        X_i = X_arr[i]
        U = np.array([model.predict_interval(x.reshape(1, -1))[1] for x in X_i])
        U_t.append(U)

    # Insert next grade predictions
    df3 = df1.copy()
    cols = df3.drop(columns=["email", "source"]).columns.to_series()
    suffixes = cols.apply(lambda x: x.split("_")[1])
    ordered_suffixes = suffixes.unique()
    prefixes2 = (
        df3.drop(columns=["email"])
        .columns.str.split("_")
        .str[:2]
        .str.join("_")
        .unique()[w1 - 1 :][::-1]
    )
    change_points = [
        i for i in range(len(cols) - 1) if suffixes[i] != suffixes[i + 1]
    ][::-1]
    for i, ut in enumerate(reversed(U_t)):
        loc = change_points[i]
        col_name = f"{prefixes2[i]}_next_grade"
        df3.insert(loc - 1, col_name, ut)

    df3["clusters"] = df2["clusters"]

    # Run analysis with clustering and next grade
    df_detail, df_agg, y_cible3, models_c_ng = run_analysis_w(
        df=df3, threshold=threshold, do_plot=False
    )
    models["+ clustering + SPCI next grade"] = models_c_ng
    print("Models with plain + clustering + SPCI next grade done")

    # SPCI last grade analysis
    models_lg = []
    INT_t = []
    k = len(X_arr)

    for i in tqdm(range(1, k), desc="Processing windows for last grade"):
        H = len(keys) - i - w2
        y = []
        for j in range(w2, len(keys)):
            if j + H < len(keys):
                y_j = Xt[keys[j + H]].iloc[:, 3]
            else:
                y_j = Xt[keys[-1]].iloc[:, 3]
            y.append(y_j)
        y_arr2 = [s.values for s in y]

        X_train = np.vstack(X_arr[:i])
        y_train = np.concatenate(y_arr2[:i])
        model = TwoSidedSPCI_RFQuant_Offline(alpha=alpha2, w=300, random_state=42)
        model.fit(X_train, y_train)
        models_lg.append(model)

        X_i = X_arr[i]
        L, U = np.array([model.predict_interval(x.reshape(1, -1))[0] for x in X_i])
        INT_t.append([L, U])
    models["SPCI last grade"] = models_lg
    print("Models for SPCI last grade done")
    mark_cols = [c for c in df3.columns if c.endswith("mark")]
    prefixes = list(dict.fromkeys(c.rsplit("_",1)[0] for c in mark_cols))
    static_cols = []

    models_comb = train_combined_models(
        dataframe=df3,
        X_arr=X_arr, 
        y_cible=y_cible3,
        X_train=X_train,
        models_c_ng=models_c_ng,
        models_lg=models_lg,
        threshold=threshold,
        w2=w2,
        prefixes=prefixes,   
        static_cols=static_cols,   
        n_estimators=500,  
        random_state=42
    )
    models["CP + SPCI last grade combined"] = models_comb


    print("Model production pipeline completed!")
    root = Path(__file__).resolve().parent
    output_dir = root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    with open(output_dir / "models.json", "w", encoding="utf-8") as f:
        json.dump(models, f, indent=4, ensure_ascii=False)

    print("✅ Dictionnaire de modèles enregistré dans root/models/models.json")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the model production pipeline for dropout prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Academic year for data processing.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="Minimum size for clusters.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="Path to the input data CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for analysis.",
    )
    parser.add_argument(
        "--w1",
        type=int,
        default=3,
        help="Window size for first analysis.",
    )
    parser.add_argument(
        "--w2",
        type=int,
        default=10,
        help="Window size for second analysis.",
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=0.1,
        help="Alpha parameter for first SPCI model.",
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=0.1,
        help="Alpha parameter for second SPCI model.",
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        main(
            year=args.year,
            n_clusters=args.n_clusters,
            min_cluster_size=args.min_cluster_size,
            data_file=args.data_file,
            threshold=args.threshold,
            w1=args.w1,
            w2=args.w2,
            alpha1=args.alpha1,
            alpha2=args.alpha2,
        )
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise
