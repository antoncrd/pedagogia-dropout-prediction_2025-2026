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

###
# SPCI next grade
w = 3
H = 0
Xt, keys, X_arr, y_arr = build_umap_windows_by_suffix(df1, w=w, H=H, target_col_idx=3, verbose=True)

U_t = []
# On parcourt i de 1 à len(X_arr)-1 (i=0 n'a pas de passé pour entraîner)
for i in tqdm(range(1, len(X_arr) - 1), desc="Fenêtres en ligne"):
    # --- 1) Construction du train sur les fenêtres passées ---
    X_train = np.vstack(X_arr[:i])      # fenêtres 0..i-1
    y_train = np.concatenate(y_arr[:i])
    # --- 2) Entraînement d'un nouveau modèle ---
    model = OneSidedSPCI_LGBM_Offline(alpha=0.1, w=300, random_state=0)
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
prefixes2 = df3.drop(columns=['email']).columns.str.split("_").str[:2].str.join("_").unique()[w-1:][::-1]
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
###

