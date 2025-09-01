from pathlib import Path
from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier
from mapie.mondrian import MondrianCP
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score
)
from sklearn.ensemble import GradientBoostingRegressor
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone

def run_analysis_w(
    csv_file: Path | None = None,
    df: pd.DataFrame | None = None,
    y: Union[pd.Series, np.ndarray] | None = None,
    *,
    alpha: float = 0.05,
    n_rendus: int = 3,
    quantile_cut: float = 0.15,
    threshold:float = None,
    nan_fill: float = 0,
    do_plot: bool = False,
    globe: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[tuple[str, int, str], MapieClassifier]]:

    # 0) Chargement et préparation
    if df is None:
        if csv_file is None:
            raise ValueError("Either df or csv_file must be provided.")
        df0 = pd.read_csv(csv_file)
    else:
        df0 = df.copy()

    if "email" not in df0.columns:
        raise ValueError("Must contain an 'email' column.")

    df0 = df0.fillna(nan_fill).rename(columns={"email": "student_id"})
    if y is None:
        # Extraction des dernières notes et normalisation
        mark_cols = [c for c in df0.columns if c.endswith("mark")][::-1]
        def last_marks(row):
            vals, cols = [], []
            for c in mark_cols:
                v = row[c]
                if v > 0:
                    vals.append(v)
                    cols.append(c)
                    if len(vals) == n_rendus:
                        break
            return pd.Series({"cols": cols, "vals": vals})

        tmp = df0.apply(last_marks, axis=1)
        df0[["last_cols", "last_vals"]] = tmp
        df0["lastvals"] = df0["last_vals"].apply(lambda r: r[0])

        if threshold is None:
            thresh = df0["lastvals"].quantile(quantile_cut)
            y_all = (df0["lastvals"] < thresh).astype(int)
            print(
            f"Quantile {quantile_cut:.2f} = {thresh:.3f} → {y_all.mean() * 100:.1f}% positives"
            )
        else:
            y_all = (df0["lastvals"] < threshold).astype(int)
            print(
            f"threshold  = {threshold:.3f} → {y_all.mean() * 100:.1f}% positives"
            )
    else:
        # si y fourni, on s'assure que c'est une Series alignée
        y_all = pd.Series(y).reset_index(drop=True).astype(int)
    y_all_arr = y_all.values
    # Features & clusters
    has_cluster = "cluster" in df0.columns
    clusters_all = df0.get("cluster", pd.Series(0, index=df0.index)).astype(int).values
    if y is None:
        prefixes = list(dict.fromkeys(c.rsplit("_",1)[0] for c in mark_cols[::-1]))
        static_cols = []
    else:
        item_cols = [c for c in df0.columns if c.startswith("Item")]

        # 2) Définir les “préfixes” comme étant ces noms de colonnes
        #    (on parcourt ensuite 1 à len(prefixes) pour ajouter les items un à un)
        prefixes = item_cols.copy()
        static_cols = [
        c for c in df0.columns
        if c not in item_cols + ["student_id", "email", "dropout", "source", "cluster"]
        ]
    def build_X(df_sub: pd.DataFrame, prefixes: list, static_cols: list, n: int) -> np.ndarray:
        # on garde student_id + les n premiers items
        dyn_cols = [
        col for col in df_sub.columns
        if any(col.startswith(pref) for pref in prefixes[:n])
        ]
        keep = ["student_id"] + static_cols + dyn_cols
        return df_sub[keep].set_index("student_id").values

    # Classifiers
    MODELS = {
        "RF":  RandomForestClassifier(n_estimators=1000, random_state=42),# RandomForestClassifier(n_estimators=1000, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced",
                                 n_jobs=-1, random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }

    records: list[dict] = []
    trained_clfs: Dict[tuple[str,int,str], Union[MapieClassifier, MondrianCP]] = {}

    for name, base_clf in MODELS.items():
        for n in tqdm(range(1, len(prefixes) + 1), desc=name):
            clf = clone(base_clf)
            idx_all = df0.index.values
            idx_tmp, idx_test, y_tmp, y_test, cl_tmp, cl_test = train_test_split(
                idx_all, y_all_arr, clusters_all,
                test_size=0.20, stratify=y_all, random_state=42
            )
            idx_tr, idx_cal, y_tr, y_cal, cl_tr, cl_cal = train_test_split(
                idx_tmp, y_tmp, cl_tmp,
                test_size=0.20/0.80, stratify=y_tmp, random_state=42
            )
            # print(df0.loc[idx_tr].columns)
            X_tr   = build_X(df0.loc[idx_tr], prefixes,static_cols,  n)
            X_cal  = build_X(df0.loc[idx_cal], prefixes, static_cols, n)
            X_test = build_X(df0.loc[idx_test], prefixes, static_cols, n)

            # Calibration globale (vanilla CP)
            clf.fit(X_tr, y_tr)
            calib = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
            calib.fit(X_cal, y_cal)
            mapie_global  = MapieClassifier(estimator=calib, method="lac", cv="prefit").fit(X_cal, y_cal)
            base_mapie    = MapieClassifier(estimator=calib, method="lac", cv="prefit")
            # Mondrian CP
            ser = pd.Series(y_cal, index=pd.Index(cl_cal, name="cluster"))
            valid_clusters = ser.groupby("cluster").nunique().loc[lambda s: s>=2].index.values
            # print('valid clusters : ', valid_clusters)
            mask_cal_valid = np.isin(cl_cal, valid_clusters)
            mond_mapie = MondrianCP(mapie_estimator=base_mapie)
            mond_mapie.fit(
                X_cal[mask_cal_valid], y_cal[mask_cal_valid],
                partition=cl_cal[mask_cal_valid]
            )

            # Prédictions
            mask_real = df0.loc[idx_test, "source"] == "real"
            # Vanilla CP sur tout
            yp_van, yps_van = mapie_global.predict(X_test, alpha=alpha)
            pset_van = yps_van[:, :, 0]
            cov_van_all   = classification_coverage_score(y_test[mask_real], pset_van[mask_real])
            width_van_all = classification_mean_width_score(pset_van[mask_real])
            records.append({
                "method": "vanilla",
                "model": name,
                "n_projects": n,
                "cluster": -1,
                "coverage": cov_van_all,
                "width": width_van_all
            })
            if has_cluster:
                for cl in np.unique(cl_test):
                    mask_cl = (cl_test == cl) & mask_real
                    if not mask_cl.any():
                        continue
                    cov_k   = classification_coverage_score(y_test[mask_cl], pset_van[mask_cl])
                    width_k = classification_mean_width_score(pset_van[mask_cl])
                    records.append({
                        "method": "vanilla",
                        "model": name,
                        "n_projects": n,
                        "cluster": cl,
                        "coverage": cov_k,
                        "width": width_k
                    })

            # Mondrian sur clusters valides et fallback
            yp_mon = np.empty_like(y_test)
            pset_mon = np.empty((len(y_test), 2), dtype=bool)
            mask_valid   = np.isin(cl_test, valid_clusters)
            mask_invalid = ~mask_valid
            if mask_valid.any():
                try:
                    yp_v, yps_v = mond_mapie.predict(
                        X_test[mask_valid], alpha=alpha,
                        partition=cl_test[mask_valid]
                    )
                except ValueError:
                    yp_v, yps_v = mapie_global.predict(X_test[mask_valid], alpha=alpha)
                yp_mon[mask_valid]  = yp_v
                pset_mon[mask_valid] = yps_v[:, :, 0]
            if mask_invalid.any():
                if globe:
                    yp_g, yps_g = mapie_global.predict(X_test[mask_invalid], alpha=alpha)
                    yp_mon[mask_invalid]  = yp_g
                    pset_mon[mask_invalid] = yps_g[:, :, 0]
                else:
                    for k in np.unique(cl_test[mask_invalid]):
                        mask_k = mask_invalid & (cl_test == k)
                        y_k = np.unique(y_cal[cl_cal == k])[0]
                        yp_mon[mask_k] = y_k
                        pset_bool_k = np.zeros((mask_k.sum(), pset_mon.shape[1]), dtype=bool)
                        pset_bool_k[:, int(y_k)] = True
                        pset_mon[mask_k] = pset_bool_k

            cov_mon_all   = classification_coverage_score(y_test[mask_real], pset_mon[mask_real])
            width_mon_all = classification_mean_width_score(pset_mon[mask_real])
            records.append({
                "method": "mondrian",
                "model": name,
                "n_projects": n,
                "cluster": -1,
                "coverage": cov_mon_all,
                "width": width_mon_all
            })
            if has_cluster:
                for cl in np.unique(cl_test):
                    mask_cl = (cl_test == cl) & mask_real
                    if not mask_cl.any():
                        continue
                    cov_k   = classification_coverage_score(y_test[mask_cl], pset_mon[mask_cl])
                    width_k = classification_mean_width_score(pset_mon[mask_cl])
                    records.append({
                        "method": "mondrian",
                        "model": name,
                        "n_projects": n,
                        "cluster": cl,
                        "coverage": cov_k,
                        "width": width_k
                    })

            # Stockage des estimateurs
            trained_clfs[(name, n, "vanilla")]   = mapie_global
            trained_clfs[(name, n, "mondrian")] = mond_mapie

    # Agrégation
    df_detail = pd.DataFrame.from_records(records)
    df_agg_cluster = (
        df_detail.groupby(["method", "model", "cluster"]).agg(
            mean_coverage=("coverage", "mean"),
            mean_width=("width", "mean")
        ).reset_index()
    )
    df_agg_global = (
        df_detail.groupby(["method", "model"]).agg(
            mean_coverage=("coverage", "mean"),
            mean_width=("width", "mean")
        ).reset_index()
        .assign(cluster="ALL")
    )
    df_agg = pd.concat([df_agg_cluster, df_agg_global], ignore_index=True)

    if do_plot:
        for metric, label in [("coverage", "Couverture"), ("width", "Width")]:
            for method in df_detail["method"].unique():
                for c in df_detail["cluster"].fillna("ALL").unique():
                    mask = (df_detail["method"] == method) & (
                        (df_detail["cluster"] == c) if c != "ALL" else df_detail["cluster"].isna() | (df_detail["cluster"] == None)
                    )
                    if not mask.any():
                        continue
                    plt.figure()
                    for model, grp in df_detail[mask].groupby("model"):
                        grp_sorted = grp.sort_values("n_projects")
                        plt.plot(grp_sorted["n_projects"], grp_sorted[metric], label=model, marker="o", linewidth=2)
                    plt.xlabel("Nombre de projets")
                    plt.ylabel(label)
                    plt.title(f"{label} ({method}, α={alpha}) – cluster {c}")
                    plt.grid(True)
                    plt.legend(); plt.tight_layout(); plt.show()

    return df_detail, df_agg, y_all, trained_clfs

class OneSidedSPCI_LGBM_Offline:
    """
    SPCI unilatéral [0 ; U] entraîné *hors-ligne*.

    • RandomForestRegressor  → prédiction ponctuelle f̂
    • LightGBM (quantile, alpha=1-α) → Q̂_t(1-α) calculé sur une fenêtre
      fixe de résidus (longueur w) dérivés du jeu d'apprentissage.
    """

    # ---------- initialisation ----------
    def __init__(self, alpha=0.1, w=400,
                 n_estimators=200, max_depth=-1,
                 random_state=0):

        self.alpha = alpha
        self.w     = w
        self.res_buf = deque()

        self.base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state,
        )

        self.gbr_upper = LGBMRegressor(
            objective="quantile",
            alpha=1 - alpha,
            n_estimators=600,          # plus d’arbres
            learning_rate=0.05,        # LR plus petit
            num_leaves=15,             # plus de souplesse
            min_data_in_leaf=5,        # autoriser petits nœuds
            max_depth=5,
            random_state=random_state,
        )
        self.gbr = GradientBoostingRegressor(
            loss="quantile",
            alpha=1 - alpha,    # quantile souhaité
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=5,
            learning_rate=0.05,
            random_state=0
        )
        self.is_ready = False                 # devient True après fit()

    # ---------- entraînement offline ----------
    def fit(self, X_train, y_train):
        """Apprend f̂ et Q̂(1-α) une seule fois."""
        # 1) f̂
        self.base_rf.fit(X_train, y_train)
        print("fit 1 ok")
        # 2) résidus tronqués à 0 -> tampon
        r = np.maximum(0.0, y_train - self.base_rf.predict(X_train))
        self.res_buf.extend(r)

        # 3) fenêtre glissante -> apprentissage du quantile
        if len(self.res_buf) < self.w + 5:
            raise ValueError("Jeu d'entraînement trop court pour w={}".format(self.w))

        R = np.asarray(self.res_buf, dtype=float)
        Y = R[self.w:]                                       # cibles r_{t'}
        X = np.array([R[i - self.w:i] for i in range(self.w, len(R))])

        self.gbr.fit(X, Y)
        print("fit 2 ok")
        self.is_ready = True
        return self

    # ---------- prédiction ----------
    def predict_interval(self, x_t):
        """
        Renvoie l'intervalle [0 ; U_t] sans mise à jour.
        """
        if not self.is_ready:
            raise RuntimeError("Le modèle doit être entraîné via .fit() avant prédiction.")

        x_t = np.asarray(x_t).reshape(1, -1)
        y_hat = self.base_rf.predict(x_t)[0]

        x_feat = np.array(self._window_features()).reshape(1, -1)
        q_sup  = self.gbr.predict(x_feat)[0]
        U_t    = max(0.0, y_hat + q_sup)

        return 0.0, U_t

    # ---------- outil interne ----------
    def _window_features(self):
        """Renvoie la fenêtre de résidus utilisée à l'entraînement (longueur w)."""
        buf = list(self.res_buf)
        if len(buf) < self.w:                    # ne devrait jamais arriver après .fit()
            buf = [0.0] * (self.w - len(buf)) + buf
        return buf[-self.w:]
    
class TwoSidedSPCI_RFQuant_Offline:
    """
    Intervalle bilatéral SPCI (équations 10‑11) estimé hors‑ligne :

        [  f̂(Xₜ) + Q̂_{b,t}(β̂) ,  f̂(Xₜ) + Q̂_{b,t}(1‑α+β̂)  ]
        β̂ = argmin_{β∈[0,α]} ( Q̂_{b,t}(1‑α+β) − Q̂_{b,t}(β) )

    Les quantiles Q̂ sont prédits par une forêt de régression quantile
    entraînée sur les w derniers résidus.

    Hyp. : scikit‑learn ≥ 1.0 (mais même sans l’API quantile native,
           on calcule les quantiles en agrégeant les prédictions
           individuelles des arbres comme le fait scikit‑garden).
    """

    def __init__(self,
                 alpha=0.10,
                 w=400,
                 n_estimators=200,
                 n_estimators_q=300,
                 max_depth=-1,
                 random_state=0,
                 n_grid=101):
        self.alpha      = float(alpha)
        self.w          = int(w)
        self.n_grid     = int(n_grid)

        # --- 1) modèle ponctuel f̂
        self.base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state,
        )

        # --- 2) modèle quantile sur les résidus
        self.rf_quant = RandomForestRegressor(
            n_estimators=n_estimators_q,
            max_depth=None if max_depth == -1 else max_depth,
            random_state=random_state + 1,
        )

        self.res_buf  = deque()   # buffer circulaire des résidus
        self.is_ready = False

    # ------------------------------------------------------------------ #
    #  Apprentissage                                                     #
    # ------------------------------------------------------------------ #
    def fit(self, X_train, y_train):
        """Entraîne le modèle ponctuel puis la forêt quantile."""
        # 1) f̂
        self.base_rf.fit(X_train, y_train)
        residuals = y_train - self.base_rf.predict(X_train)
        self.res_buf.extend(residuals)

        if len(self.res_buf) < self.w + 5:
            raise ValueError(f"Besoin d’au moins w+5={self.w+5} résidus ; "
                             f"buffer={len(self.res_buf)}")

        # 2) fenêtre glissante des résidus (caractéristiques) ────────
        R  = np.asarray(self.res_buf, dtype=float)
        Y  = R[self.w:]                                       # cible = r_t'
        Xq = np.array([R[i-self.w:i] for i in range(self.w, len(R))])

        # 3) forêt de régression quantile
        self.rf_quant.fit(Xq, Y)

        self.is_ready = True
        return self

    # ------------------------------------------------------------------ #
    #  Méthodes utilitaires                                              #
    # ------------------------------------------------------------------ #
    def _window_residuals(self):
        """Renvoie les w derniers résidus (paddés à 0 si nécessaire)."""
        buf = list(self.res_buf)
        if len(buf) < self.w:
            buf = [0.0] * (self.w - len(buf)) + buf
        return np.asarray(buf[-self.w:], dtype=float).reshape(1, -1)

    @staticmethod
    def _rf_quantile(tree_preds, q):
        """Quantile empirique des prédictions des arbres (0 ≤ q ≤ 1)."""
        return float(np.quantile(tree_preds, q, method="linear"))

    # ------------------------------------------------------------------ #
    #  Prédiction d’intervalle                                           #
    # ------------------------------------------------------------------ #
    def predict_interval(self, x_t):
        """
        Renvoie (Lₜ, Uₜ) selon SPCI, avec β̂ choisi pour largeur minimale.
        """
        if not self.is_ready:
            raise RuntimeError("Appeler .fit() avant predict_interval().")

        # 1) prédiction ponctuelle
        y_hat = self.base_rf.predict(np.asarray(x_t).reshape(1, -1))[0]

        # 2) représentation de la fenêtre courante
        X_win = self._window_residuals()

        # 3) prédictions de tous les arbres de la forêt quantile
        tree_preds = np.array([
            est.predict(X_win)[0] for est in self.rf_quant.estimators_
        ])

        # 4) balayage de β ∈ [0, α] sur une grille uniforme
        betas = np.linspace(0.0, self.alpha, self.n_grid)
        best_width = np.inf
        best_low = best_up = 0.0

        for beta in betas:
            q_low = self._rf_quantile(tree_preds, beta)
            q_up  = self._rf_quantile(tree_preds, 1.0 - self.alpha + beta)
            width = q_up - q_low
            if width < best_width:
                best_width, best_low, best_up = width, q_low, q_up

        L_t = y_hat + best_low
        U_t = y_hat + best_up
        return L_t, U_t

def build_X(df_sub: pd.DataFrame, prefixes: list, static_cols: list, n: int) -> np.ndarray:
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