#!/usr/bin/env python3
"""
Agrège verticalement les CSV d’un dossier :

- Le « préfixe » est tout ce qui précède le dernier “_”.
  ex.  B-CPE-110_secured_TSE-3-2.csv  →  B-CPE-110_secured
- Tous les fichiers partageant ce préfixe sont concaténés
  dans <outdir>/<prefix>.csv.

Options :
    --indir   dossier source      (défaut : .)
    --outdir  dossier destination (défaut : agg)
    --filter  motif shell *joker* (facultatif)
"""
from pathlib import Path
import argparse
import fnmatch
import pandas as pd
import re
from collections import defaultdict

# ────────────── helpers ────────────── #
def extract_city_from_filename(name: str) -> str:
    stem = Path(name).stem                     # ex. "B-CPE-100_star_NCY-1-1"
    if "_" not in stem:
        return ""
    suffix = stem.rsplit("_", 1)[-1]            # ex. "NCY-1-1"
    token = suffix.split("-", 1)[0]            # ex. "NCY"
    return token.strip().upper()

def add_city_column(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Ajoute/complète la colonne 'city'. L’insère après 'email' si présent."""
    if "city" not in df.columns:
        if "email" in df.columns:
            pos = df.columns.get_loc("email") + 1
            df.insert(pos, "city", city)
        else:
            df["city"] = city
    else:
        df["city"] = df["city"].fillna(city)
    return df

# ────────────── CLI ────────────── #
ap = argparse.ArgumentParser(description="Agrégation verticale de CSV par préfixe")
ap.add_argument("--indir",  type=Path, default=Path("."), help="Dossier contenant les CSV")
ap.add_argument("--outdir", type=Path, default=Path("agg"), help="Dossier de sortie")
ap.add_argument("--filter", type=str,  default="*.csv",
                help='Filtre style shell (ex. "B-CPE-110_*"). Défault: *.csv')
args = ap.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

# ────────── collecte & groupage ────────── #
groups: dict[str, list[Path]] = defaultdict(list)

for csv_path in args.indir.rglob("*.csv"):
    if not fnmatch.fnmatch(csv_path.name, args.filter):
        continue
    prefix = csv_path.stem.rsplit("_", 1)[0]   # tout avant le dernier "_"
    groups[prefix].append(csv_path)

if not groups:
    print("Aucun fichier correspondant au filtre.")
    raise SystemExit

# ────────── concaténation & écriture ────── #
for prefix, files in groups.items():
    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        city = extract_city_from_filename(f.name)
        df = add_city_column(df, city)
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    out_path = args.outdir / f"{prefix}.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"✅  {prefix}.csv : {len(files)} fichier(s) ➜ {df_combined.shape[0]} lignes")

print(f"\nTerminé : CSV agrégés dans «{args.outdir.resolve()}».")
