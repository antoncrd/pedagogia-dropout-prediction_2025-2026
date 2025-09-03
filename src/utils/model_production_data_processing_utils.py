from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Any
import umap
from sklearn.preprocessing import MinMaxScaler


def cluster_with_min_size(
    df1: pd.DataFrame,
    X: np.ndarray | pd.DataFrame,
    *,
    n_clusters: int,
    min_cluster_size: int = 50,
    random_state: int = 42,
    cluster_col: str = "cluster",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    K-means clustering avec post-traitement pour forcer une taille minimale par cluster.
    Les points appartenant aux clusters trop petits sont réaffectés à la centroïde valide la plus proche.

    Paramètres
    ----------
    df1 : DataFrame d'origine (copiée avant ajout de la colonne cluster).
    X   : array-like (n_samples, n_features) utilisé pour le clustering.
    n_clusters : nombre de clusters pour KMeans.
    min_cluster_size : taille minimale désirée pour chaque cluster.
    random_state : graine pour la reproductibilité.
    cluster_col : nom de la colonne de sortie contenant les labels de cluster.
    verbose : si True, imprime des infos de progression et de tailles.

    Retourne
    --------
    df2 : DataFrame = df1.copy() avec la colonne `cluster_col`.
    info : dict contenant:
        - 'scaler' : StandardScaler ajusté
        - 'kmeans' : modèle KMeans ajusté (avant réaffectations)
        - 'centroids' : centroïdes initiales de KMeans (np.ndarray)
        - 'clusters' : np.ndarray final des labels (après réaffectations)
        - 'sizes' : Series des tailles finales par cluster
        - 'small_clusters' : liste des clusters initialement trop petits
    """
    # 1) Standardisation + KMeans initial
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df2 = df1.copy()
    df2[cluster_col] = clusters

    if verbose:
        print("DF plain + clustering done")
        print("nombre d'élèves par cluster :")
        print(pd.Series(clusters).value_counts().sort_index())

    # 2) Post-traitement : réaffectation des points de petits clusters
    centroids = kmeans.cluster_centers_
    sizes = pd.Series(clusters).value_counts().sort_index()
    small_clusters: List[int] = sizes[sizes < min_cluster_size].index.tolist()

    if small_clusters:
        if verbose:
            print(f"Clusters trop petits à réaffecter : {small_clusters}")

        # Pour chaque point d'un cluster trop petit, déplacer vers la centroïde valide la plus proche
        for sc in small_clusters:
            idxs = np.where(clusters == sc)[0]
            for i in idxs:
                dists = np.linalg.norm(X_scaled[i] - centroids, axis=1)
                dists[sc] = np.inf  # interdire de rester dans le même (petit) cluster
                clusters[i] = int(np.argmin(dists))

        # Recalcul des tailles après réaffectations
        sizes = pd.Series(clusters).value_counts().sort_index()
        df2[cluster_col] = clusters

        if verbose:
            print("Nouvelles tailles de clusters :")
            print(sizes)

    info: Dict[str, Any] = {
        "scaler": scaler,
        "kmeans": kmeans,
        "centroids": centroids,
        "clusters": clusters,
        "sizes": sizes,
        "small_clusters": small_clusters,
    }
    return df2, info



def build_umap_windows_by_suffix(
    df1: pd.DataFrame,
    *,
    id_col: str = "email",
    test_suffix: str = "passed",
    # UMAP
    n_components: int = 3,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    metric: str = "hamming",
    random_state: int = 42,
    # Fenêtrage
    w: int = 3,           # taille de fenêtre (nb d'étapes passées)
    H: int = 0,           # horizon pour la cible (0 = même étape, 1 = étape suivante, etc.)
    target_col_idx: int = 3,  # indice de la colonne à prendre comme y dans Xt[suffix]
    verbose: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[np.ndarray], List[np.ndarray]]:
    """
    1) Regroupe les colonnes par suffixe (après le premier '_').
    2) Pour chaque suffixe, projette en 3D avec UMAP les colonnes se terminant par `test_suffix`
       (ex: '*_passed'), normalise [0,1], puis concatène avec les colonnes restantes.
    3) Construit des features X en concaténant horizontalement les w DataFrames précédents.
       La cible y est la colonne `target_col_idx` (par défaut 3) du DataFrame à l'horizon i+H.

    Paramètres
    ----------
    df1 : DataFrame d'entrée (contient au moins une colonne 'email' et des colonnes nommées 'prefix_suffix').
    id_col : Nom de la colonne identifiant (exclue du groupement).
    test_suffix : Suffixe de colonnes tests (finissant par ce motif) utilisé pour UMAP.
    (UMAP) n_components, n_neighbors, min_dist, metric, random_state : hyperparamètres UMAP.
    w : taille de fenêtre (nb d'étapes passées à concaténer pour former X_i).
    H : horizon pour y (0 = étape courante, 1 = suivante, etc.).
    target_col_idx : indice de colonne (dans Xt[suffix]) utilisé comme cible y.
    verbose : traces console.

    Retour
    ------
    Xt : dict suffix -> DataFrame (UMAP3 normalisé + colonnes non-tests).
    keys : ordre des suffixes conservé.
    X_array_hori : liste de np.ndarray (features fenêtrées).
    y_array_hori : liste de np.ndarray (cibles alignées).
    """
    # 0) Colonnes candidate (on ignore id_col si présent)
    cols_all = df1.columns.drop(id_col, errors="ignore")

    # Conserver uniquement les colonnes contenant un '_' pour éviter les IndexError
    cols_with_us = [c for c in cols_all if "_" in c]
    if not cols_with_us:
        raise ValueError("Aucune colonne avec '_' trouvée pour extraire les suffixes.")

    # 1) Extraction des suffixes (après le premier '_') en conservant l'ordre d'apparition
    col_series = pd.Series(cols_with_us)
    suffixes = col_series.apply(lambda x: x.split("_", 1)[1])
    ordered_suffixes = suffixes.drop_duplicates().tolist()

    # 2) Groupement des colonnes par suffixe
    dfs: Dict[str, pd.DataFrame] = {}
    for suffix in ordered_suffixes:
        cols_for_suffix = [c for c in cols_with_us if c.split("_", 1)[1] == suffix]
        subdf = df1[cols_for_suffix].copy()
        dfs[suffix] = subdf
        if verbose:
            print(f"Suffixe = {suffix} → shape {subdf.shape}")

    # 3) UMAP sur tests '*_passed' puis concat avec colonnes restantes
    Xt: Dict[str, pd.DataFrame] = {}
    for suffix, subdf in dfs.items():
        test_cols = [c for c in subdf.columns if c.endswith(test_suffix)]
        if len(test_cols) == 0:
            if verbose:
                print(f"[WARN] Aucun test '{test_suffix}' pour suffixe '{suffix}', on saute UMAP (colonnes non-tests seules).")
            # Si pas de tests, on garde seulement les colonnes restantes
            Xt[suffix] = subdf.fillna(0)
            if verbose:
                print(suffix, "done")
            continue

        subdf_tests = subdf[test_cols].fillna(0)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(subdf_tests.values)

        # DataFrame UMAP indexé comme subdf (pas 'df' qui n'existe pas)
        umap_cols = [f"UMAP_{i+1}_{test_suffix}" for i in range(n_components)]
        subdf_umap = pd.DataFrame(embedding, columns=umap_cols, index=subdf.index)

        # Normalisation [0,1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        umap_norm = scaler.fit_transform(subdf_umap)
        subdf_umap_norm = pd.DataFrame(umap_norm, index=subdf_umap.index, columns=subdf_umap.columns)

        # Concat UMAP normalisé + colonnes non-tests (on reproduit ignore_index=True comme dans ton code)
        Xt[suffix] = pd.concat(
            [subdf_umap_norm, subdf.drop(columns=test_cols)],
            axis=1,
            ignore_index=True
        ).fillna(0)

        if verbose:
            print(suffix, "done")

    # 4) Construction des fenêtres X et des cibles y
    keys: List[str] = list(Xt.keys())  # insertion order = ordered_suffixes
    if len(keys) < w + 1 and verbose:
        print(f"[WARN] Pas assez d'étapes ({len(keys)}) pour une fenêtre w={w}.")

    X_frames: List[pd.DataFrame] = []
    y_series: List[pd.Series] = []

    for i in range(w, len(keys)):
        # Fenêtre des w précédents
        frames = [Xt[keys[i - j]] for j in range(1, w + 1)]
        X_frames.append(pd.concat(frames, axis=1))

        # Cible à l'horizon i+H (sécurisée)
        tgt_key_idx = min(i + H, len(keys) - 1)
        tgt_df = Xt[keys[tgt_key_idx]]
        if target_col_idx >= tgt_df.shape[1]:
            raise IndexError(
                f"target_col_idx={target_col_idx} hors limites pour Xt['{keys[tgt_key_idx]}'] "
                f"avec {tgt_df.shape[1]} colonnes."
            )
        y_series.append(tgt_df.iloc[:, target_col_idx])

    # 5) Conversion en arrays
    X_array_hori: List[np.ndarray] = [df.values for df in X_frames]
    y_array_hori: List[np.ndarray] = [s.values for s in y_series]

    return Xt, keys, X_array_hori, y_array_hori


def build_X_s(df_sub: pd.DataFrame, prefixes: list, static_cols: list, n: int) -> np.ndarray:
    # on garde student_id + les n premiers items
    dyn_cols = [
    col for col in df_sub.columns
    if any(col.startswith(pref) for pref in prefixes[:n])
    ]
    keep = ["email"] + static_cols + dyn_cols
    return df_sub[keep].set_index("email").values


def pred_sets_to_bool(pred_sets, n_classes):
    """pred_sets peut être:
       - ndarray bool/int de shape (n_samples, n_classes)
       - ndarray object, chaque item étant une liste/tuple/dict des labels présents
       Retourne un ndarray bool shape (n_samples, n_classes)
    """
    pred_sets = np.asarray(pred_sets, dtype=object)       # assure le type
    if pred_sets.ndim == 2 and pred_sets.dtype != object:
        return pred_sets.astype(bool)                     # cas déjà rectangulaire

    n_samples = pred_sets.shape[0]
    out = np.zeros((n_samples, n_classes), dtype=bool)
    for i, labels in enumerate(pred_sets):
        # labels peut être scalaire, liste, tuple, ndarray…
        if np.isscalar(labels):
            out[i, int(labels)] = True
        else:
            out[i, [int(l) for l in labels]] = True
    return out