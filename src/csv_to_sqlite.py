#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv_to_sqlite.py — Charge un CSV, filtre ses colonnes, convertit en numérique,
puis écrit dans une base SQLite.

Exemples
--------
# Strictement les colonnes qui commencent par B-CPE-100
python src/utils/csv_to_sqlite.py \
  /app/data/DATA.csv /app/data/traces_metabase.db \
  --table traces \
  --include-prefixes "B-CPE-100" \
  --if-exists replace

# Garder aussi quelques colonnes (p.ex. id numériques)
python src/utils/csv_to_sqlite.py \
  /app/data/DATA.csv /app/data/traces_metabase.db \
  --table traces \
  --include-prefixes "B-CPE-100,B-CPE-110" \
  --keep-cols "student_id,year" \
  --index "student_id" \
  --if-exists replace
"""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd
from typing import Iterable

def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def select_columns(df: pd.DataFrame, prefixes: list[str], keep_cols: list[str]) -> pd.DataFrame:
    if prefixes:
        by_prefix = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    else:
        by_prefix = list(df.columns)  # pas de filtre -> tout
    forced = [c for c in keep_cols if c in df.columns]
    cols = sorted(set(by_prefix).union(forced), key=lambda c: list(df.columns).index(c))
    return df[cols]


def coerce_numeric(df: pd.DataFrame, skip: Iterable[str] = ("email",)) -> pd.DataFrame:
    out = df.copy()
    # Colonnes à convertir (toutes sauf celles à exclure)
    to_convert = [c for c in out.columns if c not in set(skip)]
    for c in to_convert:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def create_indexes(conn: sqlite3.Connection, table: str, index_cols: list[str]) -> None:
    cur = conn.cursor()
    for col in index_cols:
        idx_name = f"idx_{table}_{col}"
        cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ("{col}");')
    conn.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path, help="Chemin du CSV en entrée")
    p.add_argument("--output_db", type=Path, help="Chemin du fichier .db (SQLite)")
    p.add_argument("--table", required=True, help="Nom de la table cible (ex: traces)")
    p.add_argument("--include-prefixes", default="", help='Liste CSV de préfixes (ex: "B-CPE-100,B-CPE-110")')
    p.add_argument("--keep-cols", default="", help='Colonnes à forcer (ex: "student_id,year")')
    p.add_argument("--index", default="", help='Colonnes à indexer (ex: "student_id,year")')
    p.add_argument("--if-exists", choices=["fail", "replace", "append"], default="fail",
                   help="Comportement si la table existe déjà (défaut: fail)")
    args = p.parse_args()

    prefixes = _split_csv(args.include_prefixes)
    keep_cols = _split_csv(args.keep_cols)
    index_cols = _split_csv(args.index)

    # 1) Lecture CSV
    df = pd.read_csv(args.input_csv)

    # 2) Filtrage colonnes par préfixe (+ keep-cols)
    df = select_columns(df, prefixes, keep_cols)

    # 3) Conversion numérique
    df = coerce_numeric(df, skip=["email"])

    # 4) Écriture SQLite
    conn = sqlite3.connect(args.output_db)
    try:
        # pandas gère: fail/replace/append
        df.to_sql(args.table, conn, if_exists=args.if_exists, index=False)
        # 5) Index
        if index_cols:
            # On ne crée d’index que pour les colonnes effectivement présentes
            index_cols = [c for c in index_cols if c in df.columns]
            create_indexes(conn, args.table, index_cols)
    finally:
        conn.close()

    # Petit récap utile
    print(f"✔ Table '{args.table}' écrite dans {args.output_db}")
    print(f"  Colonnes ({len(df.columns)}): {', '.join(df.columns)}")
    print(f"  Lignes: {len(df)}")
    if index_cols:
        print(f"  Index créés sur: {', '.join(index_cols)}")


if __name__ == "__main__":
    main()
