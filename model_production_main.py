import functions_CP2

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

year = 24
n_clusters = 4

df_ = pd.read_csv("data/DATA.csv")
mask = df_.columns.str.startswith("B-CPE-210")
df_ = df_.loc[:, ~mask]
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
    
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df2 = df1.copy()
df2['cluster'] = clusters
print("DF plain + clustering done")
print("nombre d'élèves par cluster : ", pd.Series(clusters).value_counts().sort_index())


