#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_df_to_sqlite.py — Charge un CSV en DataFrame, le traite, puis écrit en SQLite (.db)

Exemples
--------
# 1) Un CSV → table traces ; ne garder que les colonnes qui commencent par B-CPE-100 ou B-CPE-110
#    + garder 'email' & 'student_id' (même si pas de préfixe)
python csv_df_to_sqlite.py \
  /app/data/DATA.csv \
  /app/data/traces_metabase.db \
  --table traces \
  --include-prefixes "B-CPE-100,B-CPE-110" \
  --keep-cols "email,student_id" \
  --pk email \
  --index "email"

# 2) Aperçu des colonnes retenues, sans écrire dans la DB
python csv_df_to_sqlite.py /app/data/DATA.csv /app/data/out.db --table t --include-prefixes "mark_,score_" --dry-run
"""

from __future__ import annotations
import argparse
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np

# ---------------------------------
# Utils
# ---------------------------------
NULL_TOKENS_DEFAULT = {"", "na", "n/a", "nan", "null", "none", "-"}

SQLITE_KEYWORDS = {
    "ABORT","ACTION","ADD","AFTER","ALL","ALTER","ANALYZE","AND","AS","ASC","ATTACH",
    "AUTOINCREMENT","BEFORE","BEGIN","BETWEEN","BY","CASCADE","CASE","CAST","CHECK",
    "COLLATE","COLUMN","COMMIT","CONFLICT","CONSTRAINT","CREATE","CROSS","CURRENT_DATE",
    "CURRENT_TIME","CURRENT_TIMESTAMP","DATABASE","DEFAULT","DEFERRABLE","DEFERRED",
    "DELETE","DESC","DETACH","DISTINCT","DROP","EACH","ELSE","END","ESCAPE","EXCEPT",
    "EXCLUSIVE","EXISTS","EXPLAIN","FAIL","FOR","FOREIGN","FROM","FULL","GLOB","GROUP",
    "HAVING","IF","IGNORE","IMMEDIATE","IN","INDEX","INDEXED","INITIALLY","INNER",
    "INSERT","INSTEAD","INTERSECT","INTO","IS","ISNULL","JOIN","KEY","LEFT","LIKE",
    "LIMIT","MATCH","NATURAL","NO","NOT","NOTNULL","NULL","OF","OFFSET","ON","OR",
    "ORDER","OUTER","PLAN","PRAGMA","PRIMARY","QUERY","RAISE","RECURSIVE","REFERENCES",
    "REGEXP","REINDEX","RELEASE","RENAME","REPLACE","RESTRICT","RIGHT","ROLLBACK","ROW",
    "SAVEPOINT","SELECT","SET","TABLE","TEMP","TEMPORARY","THEN","TO","TRANSACTION",
    "TRIGGER","UNION","UNIQUE","UPDATE","USING","VACUUM","VALUES","VIEW","VIRTUAL",
    "WHEN","WHERE","WITH","WITHOUT"
}

def sanitize_col(name: str) -> str:
    """Nom de colonne safe: snake_case, pas de caractères spéciaux, pas mot-clé SQLite."""
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s.lower()
    if not s:
        s = "col"
    if s[0].isdigit():
        s = f"c_{s}"
    if s.upper() in SQLITE_KEYWORDS:
        s = f"{s}_col"
    return s

def dedupe(names: List[str]) -> List[str]:
    """Évite les doublons ('col', 'col_2', ...)."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def infer_sqlite_type_from_series(s: pd.Series) -> str:
    """
    Devine un type SQLite ('INTEGER' / 'REAL' / 'TEXT') à partir d'une série pandas.
    - Essaie d'abord INTEGER (tous les non-NaN sont des entiers)
    - Sinon REAL si numérique
    - Sinon TEXT
    """
    # Déjà numérique ?
    if pd.api.types.is_integer_dtype(s):
        return "INTEGER"
    if pd.api.types.is_float_dtype(s):
        return "REAL"

    # Si objet/chaîne → tentative de conversion
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().sum() == 0:
        return "TEXT"
    # tous les non-NaN sont des entiers ?
    frac = sn.dropna() % 1
    if (frac == 0).all():
        return "INTEGER"
    return "REAL"

def coerce_numeric_where_possible(df: pd.DataFrame, exclude: set[str] = frozenset()) -> pd.DataFrame:
    """
    Convertit les colonnes object/str en numérique quand possible (entier si possible, sinon float).
    `exclude` = colonnes à laisser TEXT.
    """
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            sn = pd.to_numeric(out[col], errors="coerce")
            if sn.notna().sum() == 0:
                # reste TEXT
                continue
            # entier si toutes les parties fractionnaires = 0
            if (sn.dropna() % 1 == 0).all():
                out[col] = sn.astype("Int64")  # entier nullable
            else:
                out[col] = sn.astype(float)
    return out

# ---------------------------------
# Core
# ---------------------------------
def build_dataframe(
    csv_path: Path,
    encoding: str,
    delimiter: Optional[str],
    include_prefixes: Optional[List[str]],
    keep_cols: Optional[List[str]],
    na_values_extra: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Lit le CSV en DataFrame, filtre les colonnes par préfixes (startswith, pandas-like),
    ajoute keep-cols, remplace les tokens nuls par NaN.
    Retourne: (df_filtré, colonnes_brutes_retenues, colonnes_sanitisées)
    """
    # Lecture brute (on laisse pandas inférer, mais on veut garder le texte tel quel)
    # On applique une liste de NA élargie.
    na_vals = sorted(set(NULL_TOKENS_DEFAULT) | set(na_values_extra or []))
    df = pd.read_csv(csv_path, encoding=encoding, sep=delimiter, na_values=na_vals, low_memory=False)



    prefixes = tuple(p for p in include_prefixes if p)
    selected_cols = [c for c in df.columns if (not prefixes or c.startswith(prefixes))]

    for name in keep_cols:
        if name in df.columns and name not in selected_cols:
            selected_cols.append(name)

    if not selected_cols:
        raise ValueError("Aucune colonne sélectionnée (préfixes/keep_cols).")
    df = df[selected_cols].copy()


    return df

def write_dataframe_to_sqlite(
    df: pd.DataFrame,
    sqlite_db: Path,
    table: str,
    if_exists: str,
    pk: Optional[str],
    index_cols: Optional[List[str]],
) -> int:
    """
    Crée la table avec schéma (types SQLite) et insère les lignes.
    Renvoie le nombre de lignes insérées.
    """
    assert if_exists in {"fail", "replace", "append"}

    # Détermination des types SQLite depuis le DF
    col_types: Dict[str, str] = {}
    for c in df.columns:
        col_types[c] = infer_sqlite_type_from_series(df[c])

    # Contrainte PK : nom déjà sanitisé (df.columns l’est)
    pk_san = sanitize_col(pk) if pk else None
    if pk_san and pk_san not in df.columns:
        raise RuntimeError(f"--pk '{pk}' → '{pk_san}' introuvable dans le DataFrame ({list(df.columns)})")

    # Nettoyage des NaN → None pour sqlite3
    df_sql = df.fillna(0)

    # Si PK défini : supprimer les lignes avec PK manquant
    if pk_san:
        missing = df_sql[pk_san].isna().sum()
        if missing > 0:
            print(f"[INFO] {missing} lignes avec PK NULL supprimées.")
            df_sql = df_sql[df_sql[pk_san].notna()]

    con = sqlite3.connect(sqlite_db)
    cur = con.cursor()
    try:
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA journal_mode = WAL;")

        if if_exists == "replace":
            cur.execute(f'DROP TABLE IF EXISTS "{table}";')
        elif if_exists == "fail":
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            if cur.fetchone():
                raise RuntimeError(f"La table '{table}' existe déjà (if_exists=fail).")

        # Création du schéma
        cols_def = []
        for c in df_sql.columns:
            t = col_types[c]
            if pk_san and c == pk_san:
                cols_def.append(f'"{c}" {t} PRIMARY KEY')
            else:
                cols_def.append(f'"{c}" {t}')
        schema = f'CREATE TABLE IF NOT EXISTS "{table}" ({", ".join(cols_def)});'
        cur.execute(schema)

        # Insertion en batch
        placeholders = ",".join(["?"] * len(df_sql.columns))
        insert_sql = f'INSERT {"OR REPLACE " if pk_san else ""}INTO "{table}" ({",".join([f"""\"{c}\"""" for c in df_sql.columns])}) VALUES ({placeholders});'

        data_iter = (tuple(row) for row in df_sql.itertuples(index=False, name=None))
        batch: List[Tuple] = []
        batch_size = 10_000
        nrows = 0
        for rec in data_iter:
            batch.append(rec)
            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                nrows += len(batch)
                batch.clear()
        if batch:
            cur.executemany(insert_sql, batch)
            nrows += len(batch)

        # Index éventuels
        if index_cols:
            for idx in index_cols:
                sc = sanitize_col(idx)
                if sc not in df_sql.columns:
                    print(f"[AVERTISSEMENT] Index ignoré: '{idx}' (→ '{sc}') absent du DF.")
                    continue
                cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_{sc}" ON "{table}" ("{sc}");')

        con.commit()
        return nrows
    finally:
        con.close()

# ---------------------------------
# CLI
# ---------------------------------
def main():
    ap = argparse.ArgumentParser(description="Charge un CSV en DataFrame, traite, puis écrit en SQLite (.db)")
    ap.add_argument("csv_path", type=Path, help="Chemin du fichier CSV")
    ap.add_argument("sqlite_db", type=Path, help="Chemin de la base SQLite de sortie (*.db)")
    ap.add_argument("--table", type=str, default=None, help="Nom de la table SQLite (défaut: nom du CSV)")
    ap.add_argument("--encoding", type=str, default="utf-8", help="Encodage du CSV (ex: utf-8, utf-8-sig)")
    ap.add_argument("--delimiter", type=str, default=None, help="Délimiteur (ex: ';'). Si absent, pandas infère.")
    ap.add_argument("--nulls", type=str, default=",".join(sorted(NULL_TOKENS_DEFAULT)),
                    help="Tokens considérés comme NULL, séparés par des virgules (ex: '',na,null,...)")
    ap.add_argument("--include-prefixes", type=str, default=None,
                    help="Préfixes de colonnes à inclure (séparés par des virgules). startswith, sensible à la casse par défaut.")
    ap.add_argument("--keep-cols", type=str, default=None,
                    help="Colonnes *brutes* à garder en plus (séparées par des virgules), ex: 'email,student_id'.")
    ap.add_argument("--ignore-case", action="store_true",
                    help="Rend le matching des préfixes et keep-cols insensible à la casse.")
    ap.add_argument("--pk", type=str, default=None,
                    help="Nom de colonne (brut) à utiliser comme PRIMARY KEY (après nettoyage → snake_case).")
    ap.add_argument("--index", type=str, default=None,
                    help="Colonnes à indexer (séparées par des virgules).")
    ap.add_argument("--if-exists", choices=["fail","replace","append"], default="replace",
                    help="Comportement si la table existe déjà.")
    ap.add_argument("--no-numeric-coerce", action="store_true",
                    help="Désactive la conversion automatique en numérique quand possible.")
    ap.add_argument("--dry-run", action="store_true", help="N’écrit pas dans la DB ; affiche les colonnes retenues.")
    args = ap.parse_args()

    include_prefixes = [p.strip() for p in (args.include_prefixes.split(",") if args.include_prefixes else []) if p.strip()]
    keep_cols = [c.strip() for c in (args.keep_cols.split(",") if args.keep_cols else []) if c.strip()]
    null_tokens = {t.strip().lower() for t in args.nulls.split(",") if t.strip() != ""}
    index_cols = [c.strip() for c in args.index.split(",")] if args.index else None

    if args.table is None:
        args.table = sanitize_col(args.csv_path.stem)

    # 1) Build DataFrame
    df = build_dataframe(
        csv_path=args.csv_path,
        encoding=args.encoding,
        delimiter=args.delimiter,
        include_prefixes=include_prefixes,
        keep_cols=keep_cols,
        na_values_extra=list(null_tokens),
    )


    # 2) Optionnel: coercition numérique
    if not args.no_numeric_coerce:
        # Exclure explicitement la PK (souvent TEXT comme email)
        exclude = {sanitize_col(args.pk)} if args.pk else set()
        df = coerce_numeric_where_possible(df, exclude=exclude)

    if args.dry_run:
        print("[DRY-RUN] Aucune écriture réalisée.")
        return

    # 3) Écriture SQLite
    args.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
    n = write_dataframe_to_sqlite(
        df=df,
        sqlite_db=args.sqlite_db,
        table=args.table,
        if_exists=args.if_exists,
        pk=args.pk,
        index_cols=index_cols,
    )
    print(f"[OK] {n} lignes insérées dans '{args.sqlite_db}' table '{args.table}'.")
    print("Terminé.")

if __name__ == "__main__":
    main()
