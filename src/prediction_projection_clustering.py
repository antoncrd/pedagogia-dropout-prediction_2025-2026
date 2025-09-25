#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de prédiction + embeddings UMAP + clustering KMeans sur la C-Pool.

Améliorations clés :
- Paramètres en ligne de commande (année, alpha, n_clusters, etc.).
- Gestion propre des chemins et de l'année courte (ex: 2025 -> 25).
- Chargement du bon fichier modèle (fallback possible).
- Sélection robuste des colonnes *_passed pour UMAP (option d’exclure celles qui contiennent 'task').
- UMAP (metric=hamming) avec coercition en bool/int.
- KMeans reproductible, cohérent avec n_clusters demandé.
- Restauration des colonnes *_mark depuis le CSV brut par merge sur 'email' (sécurisé).
- Sauvegarde dans data/ plutôt que chemin absolu.
- Logs clairs, erreurs explicites.

Usage :
    python improved_pred_umap.py --year 2025 --alpha 0.1 --max-N 1 --n-clusters 7 \
        --model-base GB --exclude-task-passed

"""

import argparse
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

# ---- Imports projet (on suppose ces fonctions disponibles) ----
from model_production_main import load_and_preprocess_data, prepare_features
from utils.model_production_data_processing_utils import (
    cluster_with_min_size,
    build_X_s,
    map_pset_to_label,
)

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------

def year_to_short(y: int) -> int:
    """Convertit l'année en format court attendu par certaines fonctions (ex: 2025 -> 25)."""
    return y % 100

def detect_repo_root() -> Path:
    """Point de départ raisonnable : répertoire courant."""
    return Path.cwd()

def code_pool_for_year(year: int) -> str:
    """Choisit le préfixe de C-Pool en fonction de l'année."""
    return "G-CPE-100" if year >= 2025 else "B-CPE-100"

def load_models(models_dir: Path, short_year: int, fallback_short_year: int | None = None):
    """
    Charge le bundle de modèles de classification conforme (MCP).
    Tente d'abord models_clustering_{short_year}.joblib,
    puis retombe sur fallback_short_year le cas échéant.
    """
    primary = models_dir / f"models_clustering_{short_year}.joblib"
    if primary.exists():
        print(f"[INFO] Chargement modèles : {primary}")
        return joblib.load(primary)

    if fallback_short_year is not None:
        fallback = models_dir / f"models_clustering_{fallback_short_year}.joblib"
        if fallback.exists():
            warnings.warn(
                f"[WARN] Modèles {primary.name} introuvables. Fallback vers {fallback.name}.",
                stacklevel=1,
            )
            return joblib.load(fallback)

    raise FileNotFoundError(
        f"Aucun fichier modèle trouvé pour {primary} "
        f"{'(ni fallback)' if fallback_short_year is None else f'ni fallback {fallback.name}'}."
    )

def select_passed_columns(df: pd.DataFrame, code_pool: str) -> list[str]:
    """
    Garde UNIQUEMENT les colonnes 'Success of taskXX_passed' du code_pool,
    et exclut les colonnes 'taskXX_passed' brutes.

    Ex:
    ✓ B-CPE-100_cpoolday01_09 - Success of task01_passed  (INCLURE)
    ✗ B-CPE-100_cpoolday01_01 - task01_passed              (EXCLURE)
    """
    # Préfixe commun attendu : "<POOL>_cpoolday<DD>_<NN> - "
    prefix = rf"^{re.escape(code_pool)}_cpoolday\d+_\d+\s+-\s+"

    success_pat = re.compile(prefix + r"Success of task\d+_passed$", re.IGNORECASE)
    raw_task_pat = re.compile(prefix + r"task\d+_passed$", re.IGNORECASE)

    # Inclure uniquement les 'Success of task..._passed'
    success_cols = [c for c in df.columns if success_pat.match(c)]

    # (Optionnel) Pour debug : colonnes brutes exclues
    excluded_raw = [c for c in df.columns if raw_task_pat.match(c)]
    if excluded_raw:
        print(f"[INFO] {len(excluded_raw)} colonnes 'taskXX_passed' exclues (brutes). Ex.: {excluded_raw[:3]}")

    if not success_cols:
        raise ValueError(
            f"Aucune colonne 'Success of taskXX_passed' trouvée pour {code_pool}. "
            f"Vérifie les noms de colonnes."
        )

    return success_cols

def ensure_bool_int_matrix(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    S'assure que la matrice est bool/int (requis pour metric='hamming').
    Convertit NaN -> 0, puis cast en int.
    """
    X = df_sub.copy()
    X = X.fillna(0)
    # Si certaines colonnes sont float {0.0,1.0}, on cast en int
    for c in X.columns:
        # Heuristique simple : si la colonne ne contient que 0/1/NaN, on int-cast
        unique_vals = pd.unique(X[c].dropna())
        if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            X[c] = X[c].astype(int)
        else:
            # Sinon, on tente un mapping booléen explicite
            X[c] = X[c].astype(bool).astype(int)
    return X

def restore_mark_columns(df_proc: pd.DataFrame, df_raw: pd.DataFrame, id_col: str = "email") -> pd.DataFrame:
    """
    Restaure les colonnes *_mark depuis le CSV brut via un merge sur id_col.
    (Plus sûr qu'un simple df[c] = df_[c] qui suppose le même ordre.)
    """
    if id_col not in df_proc.columns or id_col not in df_raw.columns:
        warnings.warn(
            f"[WARN] Impossible de restaurer *_mark : colonne id '{id_col}' absente. "
            f"Je laisse le df tel quel.",
            stacklevel=1,
        )
        return df_proc

    mark_cols = [c for c in df_raw.columns if c.endswith("mark")]
    if not mark_cols:
        warnings.warn("[WARN] Aucune colonne *_mark dans le CSV brut.", stacklevel=1)
        return df_proc

    restore = df_raw[[id_col] + mark_cols].copy()
    merged = df_proc.drop(columns=[c for c in df_proc.columns if c in mark_cols], errors="ignore") \
                   .merge(restore, on=id_col, how="left")
    return merged

# ---------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------

def run(
    year: int,
    alpha: float,
    max_N: int,
    n_clusters_umap: int,
    model_base: str,
    output_path: Path | None = None,
) -> Path:
    root = detect_repo_root()
    data_dir = root / "data"
    models_dir = root / "models"
    out_dir = data_dir

    short_year = year_to_short(year)
    code_pool = code_pool_for_year(year)

    # Chargement des données (brutes et préprocessées)
    csv_path = data_dir / f"DATA_{year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    print(f"[INFO] Chargement préprocessé : DATA_{year}.csv (year_short={short_year})")
    df = load_and_preprocess_data(csv_path, short_year)

    print(f"[INFO] Chargement brut pour restauration des *_mark : DATA_{year}.csv")
    df_raw = pd.read_csv(csv_path)

    # Features pour clustering min-size
    X_feat = prepare_features(df, year)
    df2, info = cluster_with_min_size(
        df, X_feat, n_clusters=4, min_cluster_size=50, random_state=42
    )
    # On travaillera sur df2 pour la prédiction, mais on écrira les résultats dans df_final
    df_final = df2.copy()

    # Préfixes pour build_X_s
    mark_cols = [c for c in df.columns if c.endswith("mark")]
    prefixes = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in mark_cols))
    static_cols: list[str] = []

    # Chargement des modèles conformes
    mod_CP = load_models(models_dir, short_year=short_year, fallback_short_year=24)

    # Prédictions MCP par horizon n = 1..max_N
    for n in range(1, max_N + 1):
        key = (model_base, n, "vanilla")
        if key not in mod_CP:
            warnings.warn(f"[WARN] Modèle absent dans le bundle : {key}. Skip.", stacklevel=1)
            continue

        X_CP = build_X_s(df_final.fillna(0), prefixes, static_cols, n)
        model_CP = mod_CP[key]

        # Mapie Classification: yps shape = (n_samples, n_classes, n_alpha)
        yp_van, yps_van = model_CP.predict(X_CP, alpha=alpha)  # partition=df_final['clusters'] si applicable
        if yps_van.ndim != 3 or yps_van.shape[1] != 2:
            raise ValueError(
                f"Forme inattendue de yps_van: {yps_van.shape}. "
                "Attendu (n_samples, 2 classes, n_alpha)."
            )

        pset_van_bool = yps_van[:, :, 0].astype(bool)  # (N, 2)
        labels_van = map_pset_to_label(pset_van_bool)  # ex: {0},{1},{0,1} -> {0,1,3}
        df_final[f'prediction{n}'] = labels_van

    # ---- UMAP sur colonnes *_passed de la C-Pool ----
    passed_cols = select_passed_columns(df, code_pool=code_pool)
    X_bin = ensure_bool_int_matrix(df[passed_cols])

    reducer = umap.UMAP(
        n_neighbors=8,
        min_dist=0.25,
        spread=1.0,
        n_components=2,
        metric="hamming",
        random_state=42,
    )
    print(f"[INFO] Fit UMAP sur {X_bin.shape[0]}x{X_bin.shape[1]} (metric=hamming)")
    emb = reducer.fit_transform(X_bin)

    # Clustering KMeans sur l'embedding
    print(f"[INFO] KMeans avec n_clusters={n_clusters_umap}")
    kmeans = KMeans(n_clusters=n_clusters_umap, n_init=20, random_state=42)
    labels = kmeans.fit_predict(emb)

    # Ajout au df_final
    df_final['UMAP1'] = emb[:, 0]
    df_final['UMAP2'] = emb[:, 1]
    df_final['clusters'] = labels

    # ---- Restauration sécurisée des colonnes *_mark depuis le CSV brut ----
    df_final = restore_mark_columns(df_final, df_raw, id_col="email")

    # ---- Sauvegarde ----
    out_path = output_path or (out_dir / f"DATA_{year}_pred_proj.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Fichier écrit : {out_path}")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prédiction + UMAP + KMeans C-Pool")
    p.add_argument("--year", type=int, default=2025, help="Année complète (ex: 2025).")
    p.add_argument("--alpha", type=float, default=0.10, help="Niveau alpha pour CP.")
    p.add_argument("--max-N", type=int, default=1, help="Nombre d'horizons MCP (n=1..N).")
    p.add_argument("--n-clusters", type=int, default=7, help="Clusters KMeans sur UMAP.")
    p.add_argument("--model-base", type=str, default="GB", choices=["GB"],
                   help="Base model pour la clé du bundle MCP.")
    p.add_argument("--output", type=str, default=None, help="Chemin de sortie CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run(
        year=args.year,
        alpha=args.alpha,
        max_N=args.max_N,
        n_clusters_umap=args.n_clusters,
        model_base=args.model_base,
        output_path=Path(args.output) if args.output else None,
    )
