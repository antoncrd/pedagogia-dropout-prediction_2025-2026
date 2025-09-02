import models_production_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import umap
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

year = 24
n_clusters = 4
min_cluster_size = 50
models = {}

df_ = pd.read_csv("data/DATA.csv")
df = df_.copy()
mark_cols = [c for c in df.columns if c.endswith("mark")]
df[mark_cols] = df[mark_cols].div(df[mark_cols].mean())
nb_nan_par_ligne = df.isna().sum(axis=1)
print("nombre maximum de nan dans une ligne : ", max(nb_nan_par_ligne))
if year == 24:
    df = df[nb_nan_par_ligne < 495]
elif year == 23:
    df = df[nb_nan_par_ligne < 130]

df1 = df.copy()
df1["source"] = "real"
print("DF plain done")
if year == 24:
    # CPE 2024
    dfcpool = df1[[c for c in df1.columns if c.startswith("B-CPE-100")]]
    pat = re.compile(r"B-CPE-100_cpoolday\d+_\d{2} - task\d+_passed")
    cols_keep = [c for c in dfcpool.columns if not pat.match(c)]
    dfcpool_mark = dfcpool[cols_keep]
    X = dfcpool_mark.fillna(0)
elif year == 23:
    # CPE 2023
    X = df1[[c for c in df1.columns if c.startswith("B-CPE-110_settingup")]].fillna(0)

df2, info = cluster_with_min_size(df1, X, n_clusters=n_clusters, min_cluster_size=min_cluster_size, random_state=42)

print("réaffectation des petits clusters done")

###
# Y_TARGET ??
###

df_detail, df_agg, y_cible, models_c = run_analysis_w(
        df=df2,
        threshold=threshold, # à définir
        do_plot=False,
    )

models['+ clustering'] = models_c

print("création des modèles PLAIN + clustering done")
###
# SPCI next grade
w1 = 3
H = 0
ALPHA1 = 0.1
Xt, keys, X_arr, y_arr = build_umap_windows_by_suffix(df1, w=w1, H=H, target_col_idx=3, verbose=True)

U_t = []
# On parcourt i de 1 à len(X_arr)-1 (i=0 n'a pas de passé pour entraîner)
for i in tqdm(range(1, len(X_arr) - 1), desc="Fenêtres en ligne"):
    # --- 1) Construction du train sur les fenêtres passées ---
    X_train = np.vstack(X_arr[:i])      # fenêtres 0..i-1
    y_train = np.concatenate(y_arr[:i])
    # --- 2) Entraînement d'un nouveau modèle ---
    model = OneSidedSPCI_LGBM_Offline(alpha=ALPHA1, w=300, random_state=42)
    model.fit(X_train, y_train)
    X_i, y_i = X_arr[i], y_arr[i]
    # calcul des bornes supérieures U_t pour chaque échantillon de X_i
    U = np.array([
        model.predict_interval(x.reshape(1, -1))[1]
        for x in X_i
    ])
    U_t.append(U)

df3 = df1.copy()
cols = df3.drop(columns=['email', 'source']).columns.to_series()
# 2. On extrait le "suffixe" (ce qui suit le premier '_')
suffixes = cols.apply(lambda x: x.split('_')[1])
ordered_suffixes = suffixes.unique()
prefixes2 = df3.drop(columns=['email']).columns.str.split("_").str[:2].str.join("_").unique()[w1-1:][::-1]
change_points = [i for i in range(len(cols) -1) if suffixes[i]!= suffixes[i+1]][::-1]
for i, ut in enumerate(reversed(U_t)):
    loc = change_points[i]
    col_name = f"{prefixes2[i]}_next_grade"
    df3.insert(loc - 1, col_name, ut)

df3['clusters'] = df2['clusters']

df_detail, df_agg, y_cible, models_c_ng = run_analysis_w(
        df=df3,
        threshold=threshold,
        do_plot=False)
models['+ clustering + SPCI next grade'] = models_c_ng
print("création des modèles PLAIN + clustering + SPCI next grade done")
###

models_lg = []
w2 = 10
ALPHA2 = 0.1
INT_t = []
k = len(X_arr)

# Parcours des fenêtres temporelles
for i in tqdm(range(1, k), desc="Fenêtres en ligne"):

    H = len(keys) - i - w2
    y = []

    for j in range(w2, len(keys)):
        if j+H < len(keys):
            y_j = Xt[keys[j+H]].iloc[:, 3]
        else:
            y_j = Xt[keys[-1]].iloc[:, 3]
        y.append(y_j)
    y_arr2 = [s.values for s in y]

    # 1) Construction du train sur les fenêtres passées
    X_train = np.vstack(X_arr[:i])      # fenêtres 0..i-1
    y_train = np.concatenate(y_arr2[:i])
    # 2) Entraînement d'un nouveau modèle
    model = TwoSidedSPCI_RFQuant_Offline(alpha=ALPHA2, w=300, random_state=42)
    model.fit(X_train, y_train)

    # Sauvegarde en mémoire
    models_lg.append(model)

    # 3) Évaluation sur la fenêtre courante i
    X_i = X_arr[i]
    L = np.array([
        model.predict_interval(x.reshape(1, -1))[0]
        for x in X_i
    ])
    U = np.array([
        model.predict_interval(x.reshape(1, -1))[1]
        for x in X_i
    ])
    INT_t.append([L, U])

print("création des modèles SPCI last grade")

# -----------------------------------------------------------------------------
# Configuration & constants
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42            # Ensures full reproducibility
ALPHA: float = 0.05               # Target mis-coverage level
W: int = 2                        # Sliding-window size
nan_fill = 0
threshold = threshold
loaded_models = models2sp
# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
DF = df8.copy()
DF.fillna(nan_fill, inplace=True)
DF.reset_index(drop=True, inplace=True)
prefixes = list(dict.fromkeys(c.rsplit("_",1)[0] for c in mark_cols[::-1]))
static_cols = []


# -----------------------------------------------------------------------------
# Base models
# -----------------------------------------------------------------------------
MODELS: Dict[str, object] = {
    "RF": RandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "GB": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

# -----------------------------------------------------------------------------
# Conformal prediction evaluation loop
# -----------------------------------------------------------------------------
from tqdm import tqdm

res_fin_port: List[pd.DataFrame] = []

for name, base_clf in MODELS.items():
    covs_MCP, width_MCP = [], []
    covs_SPCI, width_SPCI = [], []
    covs_comb, width_comb = [], []
    covs_union,  width_union  = [], []
    for n in tqdm(range(W, len(prefixes) - 1), desc=name):
        gate_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        # 1) split train / tmp / test
        idx_tmp, idx_test, y_tmp, y_test, cl_tmp, cl_test = train_test_split(
            DF.index, Y_TARGET, DF["cluster"],
            test_size=0.20,
            stratify=Y_TARGET,
            random_state=RANDOM_STATE,
        )
        idx_tr, idx_cal, y_tr, y_cal, cl_tr, cl_cal = train_test_split(
            idx_tmp, y_tmp, cl_tmp,
            test_size=0.40/0.80,
            stratify=y_tmp,
            random_state=RANDOM_STATE,
        )
        # **nouveau** split en deux pour CP vs gate
        idx_cal_cp, idx_cal_gate, y_cal_cp, y_cal_gate, cl_cal_cp, cl_cal_gate = train_test_split(
            idx_cal, y_cal, cl_cal,
            test_size=0.5,
            stratify=y_cal,
            random_state=RANDOM_STATE,
        )
        y_cal_cp   = np.array(y_cal_cp)
        y_cal_gate = np.array(y_cal_gate)
        y_test     = np.array(y_test)
        # mask des “réels” pour toutes les évaluations
        mask_real = DF.loc[idx_test, "source"] == "real"

        # 2) build features
        X_tr        = build_X_(DF.loc[idx_tr],       prefixes, static_cols, n)
        X_cal_cp    = build_X_(DF.loc[idx_cal_cp],   prefixes, static_cols, n)
        X_cal_gate  = build_X_(DF.loc[idx_cal_gate], prefixes, static_cols, n)
        X_test      = build_X_(DF.loc[idx_test],     prefixes, static_cols, n)
        # 3) train base clf + calibrate for MCP
        clf = clone(base_clf)
        clf.fit(X_tr, y_tr)
        calib = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid") \
                    .fit(X_cal_cp, y_cal_cp)

        base_mapie = MapieClassifier(estimator=calib, method="lac", cv="prefit")
        mond_mapie = MondrianCP(mapie_estimator=base_mapie) \
                        .fit(X_cal_cp, y_cal_cp, partition=cl_cal_cp)

        # ---- MCP on TEST ----
        _, yps_van_test = mond_mapie.predict(X_test, alpha=ALPHA, partition=cl_test)
        pset_van_test = yps_van_test[:, :, 0]
        cov_van = classification_coverage_score(y_test[mask_real], pset_van_test[mask_real])
        wid_van = classification_mean_width_score(pset_van_test[mask_real])
        covs_MCP.append(cov_van)
        width_MCP.append(wid_van)
        print("MCP", cov_van, wid_van)
        # ---- SPCI on TEST ----
        model_spci = loaded_models[n - W]
        pos_test   = DF.index.get_indexer(idx_test)
        X_spci_test = X_array_hori[n - W + 1][pos_test]
        intervals = [model_spci.predict_interval(x.reshape(1, -1))
                     for x in X_spci_test]
        L_preds, U_preds = zip(*intervals)

        y_pred_bool_SPCI = np.zeros((len(intervals), 2), dtype=bool)
        for i, (L, U) in enumerate(zip(L_preds, U_preds)):
            if threshold > U:
                y_pred_bool_SPCI[i, 1] = True
            elif threshold < L:
                y_pred_bool_SPCI[i, 0] = True
            else:
                y_pred_bool_SPCI[i, :] = True

        cov_spci = classification_coverage_score(y_test[mask_real],
                                                y_pred_bool_SPCI[mask_real])
        wid_spci = classification_mean_width_score(y_pred_bool_SPCI[mask_real])
        covs_SPCI.append(cov_spci)
        width_SPCI.append(wid_spci)
        print("SPCI", cov_spci, wid_spci)
        ##UNION
        y_pred_bool_MCP = pset_van_test.astype(bool)
        y_bool_union = y_pred_bool_MCP | y_pred_bool_SPCI
        cov_union = classification_coverage_score(
            y_test[mask_real],
            y_bool_union[mask_real]
        )
        wid_union = classification_mean_width_score(
            y_bool_union[mask_real]
        )
        covs_union.append(cov_union)
        width_union.append(wid_union)
        print("UNION :", cov_union, wid_union)

        # ---- construire la gate sur CAL_GATE ----
        #  a) MCP predictions sur X_cal_gate
        _, yps_van_gate = mond_mapie.predict(
            X_cal_gate, alpha=ALPHA, partition=cl_cal_gate
        )
        pset_cal_cls = yps_van_gate[:, :, 0]

        #  b) SPCI predictions sur X_cal_gate
        pos_cal_gate  = DF.index.get_indexer(idx_cal_gate)
        X_spci_cal    = X_array_hori[n - W + 1][pos_cal_gate]
        intervals_cal = [model_spci.predict_interval(x.reshape(1, -1))
                         for x in X_spci_cal]
        L_cal, U_cal  = zip(*intervals_cal)

        pset_cal_spc = np.zeros_like(pset_cal_cls, dtype=bool)
        for i, (L, U) in enumerate(zip(L_cal, U_cal)):
            if threshold > U:
                pset_cal_spc[i, 1] = True
            elif threshold < L:
                pset_cal_spc[i, 0] = True
            else:
                pset_cal_spc[i, :] = True

        #  c) préparer méta-features & labels pour la gate
        df_sel_arr = []
        labels_g   = []                      # ← on initialise labels_g

        for i in range(len(idx_cal_gate)):
            feat_vec = X_cal_gate[i]
            w_cls    = pset_cal_cls[i].sum()
            w_spc    = pset_cal_spc[i].sum()
            diff     = w_cls - w_spc
            err_cls  = int(y_cal_gate[i] not in np.where(pset_cal_cls[i])[0])
            err_spc  = int(y_cal_gate[i] not in np.where(pset_cal_spc[i])[0])
            if   err_cls == 0 and err_spc == 1:
                gate_y = 0
            elif err_spc == 0 and err_cls == 1:
                gate_y = 1
            elif err_cls == 0 and err_spc == 0:
                gate_y = 0 if w_cls < w_spc else 1
            else:
                gate_y = 2
            labels_g.append(gate_y)           # ← on stocke le label

            meta_vec = np.concatenate([
                 feat_vec,
                 [w_cls, w_spc, diff, err_cls, err_spc]
            ])
            df_sel_arr.append(meta_vec)

        X_gate_train = np.vstack(df_sel_arr)
        gate_clf.fit(X_gate_train, np.array(labels_g))
        # ---- appliquer la gate sur TEST ----
        meta_test_arr = []
        for i in range(len(idx_test)):
            # feat_vec est un array 1D de taille n_features
            feat_vec = X_test[i]
            w_cls = pset_van_test[i].sum()
            w_spc = y_pred_bool_SPCI[i].sum()
            diff = w_cls - w_spc
            # on concatène feat_vec et les 5 features méta
            meta_vec = np.concatenate([
                feat_vec,
                [w_cls, w_spc, diff, 0, 0]    # err_cls=0, err_spc=0
            ])
            meta_test_arr.append(meta_vec)

        # on empile en matrice (n_test × n_features_meta)
        X_gate_test = np.vstack(meta_test_arr)

        # on prédit le choix de la gate
        choices = gate_clf.predict(X_gate_test)
        pset_final = np.zeros_like(pset_van_test, dtype=bool)
        for i, choice in enumerate(choices):
            if choice == 0:
                pset_final[i] = y_pred_bool_MCP[i]
            elif choice == 1:
                pset_final[i] = y_pred_bool_SPCI[i]
            else:
                pset_final[i] = y_pred_bool_MCP[i] | y_pred_bool_SPCI[i]

        cov_c = classification_coverage_score(y_test[mask_real],
                                             pset_final[mask_real])
        wid_c = classification_mean_width_score(pset_final[mask_real])
        covs_comb.append(cov_c)
        width_comb.append(wid_c)
        print("COMBINED", cov_c, wid_c)
    # on agrège les métriques
    n_vals = list(range(W, W + len(covs_MCP)))
    df_metrics = pd.DataFrame({
        "model":             [name] * len(n_vals),
        "n":                 n_vals,
        "coverage_MCP":      covs_MCP,
        "width_MCP":         width_MCP,
        "coverage_SPCI":     covs_SPCI,
        "width_SPCI":        width_SPCI,
        "coverage_union":    covs_union,
        "width_union":       width_union,
        "coverage_combined": covs_comb,
        "width_combined":    width_comb,
    })
    res_fin_port.append(df_metrics)
