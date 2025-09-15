#!/usr/bin/env python3
"""
csv_to_sqlite.py — Convertit un (ou plusieurs) CSV en base SQLite (.db)

Exemples
--------
# Un fichier → une table `traces` (remplace si elle existe)
python csv_to_sqlite.py data/DATA.csv data/traces_metabase.db --table traces --if-exists replace --index email

# Tous les CSV d'un dossier → une table par fichier (nom = nom du fichier)
python csv_to_sqlite.py data/ data/traces_metabase.db --if-exists replace

# Délimiteur explicite et encodage
python csv_to_sqlite.py data/notes.csv data/app.db --delimiter ";" --encoding "utf-8-sig"
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

# -----------------------
# Utilitaires
# -----------------------
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
    """Transforme un nom de colonne en snake_case sûr pour SQLite."""
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s, flags=re.UNICODE)
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
    seen = {}
    out = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def sniff_dialect(sample: str, fallback_delimiter: Optional[str]) -> csv.Dialect:
    if fallback_delimiter:
        class Simple(csv.Dialect):
            delimiter = fallback_delimiter
            quotechar = '"'
            escapechar = None
            doublequote = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
            skipinitialspace = True
        return Simple
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",",";","\t","|"])
        dialect.skipinitialspace = True
        return dialect
    except Exception:
        # défaut robuste
        class Default(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            escapechar = None
            doublequote = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
            skipinitialspace = True
        return Default

def infer_type(value: str) -> str:
    """Renvoie 'INTEGER', 'REAL' ou 'TEXT' pour une valeur (chaîne)."""
    v = value.strip()
    if v == "":
        return "NULL"
    # INTEGER
    if re.fullmatch(r"[+-]?\d+", v):
        return "INTEGER"
    # REAL
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?", v) or re.fullmatch(r"[+-]?\d+[eE][+-]?\d+", v):
        return "REAL"
    return "TEXT"

def merge_types(t1: str, t2: str) -> str:
    """Hiérarchie: TEXT > REAL > INTEGER > NULL"""
    order = {"NULL":0, "INTEGER":1, "REAL":2, "TEXT":3}
    return t1 if order[t1] >= order[t2] else t2

def normalize_null(v: str, null_tokens: set[str]) -> Optional[str]:
    return None if v.strip().lower() in null_tokens else v

def cast_for_type(v: Optional[str], t: str):
    if v is None or v == "":
        return None
    if t == "INTEGER":
        try:
            return int(v)
        except Exception:
            return None
    if t == "REAL":
        try:
            return float(v)
        except Exception:
            return None
    return v

# -----------------------
# Import d'un CSV
# -----------------------
def import_csv_to_table(
    csv_path: Path,
    con: sqlite3.Connection,
    table: Optional[str],
    if_exists: str,
    encoding: str,
    delimiter: Optional[str],
    null_tokens: set[str],
    sample_rows_for_infer: int = 5000,
    create_indexes: Optional[List[str]] = None,
    pk: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Importe csv_path → table dans con. Renvoie (table_name, nb_rows).
    """
    assert if_exists in {"fail","replace","append"}
    # Déterminer nom de table
    table_name = table or sanitize_col(csv_path.stem)

    # Lecture du sample pour sniffer le dialecte et la ligne d'en-tête
    with csv_path.open("r", encoding=encoding, newline="") as f:
        sample = f.read(4096)
    dialect = sniff_dialect(sample, delimiter)

    # Première passe: header + inférence de types
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        raw_header = next(reader)
        cols = [sanitize_col(h) for h in raw_header]
        cols = dedupe(cols)

        # inférence sur échantillon
        types = ["NULL"] * len(cols)
        count = 0
        for row in reader:
            count += 1
            for i, cell in enumerate(row[:len(cols)]):
                t = infer_type(cell)
                types[i] = merge_types(types[i], t)
            if count >= sample_rows_for_infer:
                break

    # Remplace les "NULL" restants (colonnes vides au début) par TEXT
    types = [("TEXT" if t == "NULL" else t) for t in types]

    cur = con.cursor()
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")

    # Création de table
    if if_exists == "replace":
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    elif if_exists == "fail":
        # Vérifie si table existe déjà
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if cur.fetchone():
            raise RuntimeError(f"La table '{table_name}' existe déjà (if_exists=fail).")

    # Construire schéma
    cols_defs = []
    for c, t in zip(cols, types):
        if pk and c == pk:
            cols_defs.append(f'"{c}" {t} PRIMARY KEY')
        else:
            cols_defs.append(f'"{c}" {t}')
    schema = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols_defs)});'
    cur.execute(schema)

    # Deuxième passe: insertion de *toutes* les lignes
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, dialect)
        header = next(reader)  # skip
        placeholders = ",".join(["?"]*len(cols))
        insert_sql = f'INSERT {"OR REPLACE " if pk else ""}INTO "{table_name}" ({",".join([f"""\"{c}\"""" for c in cols])}) VALUES ({placeholders});'

        batch, nrows = [], 0
        batch_size = 10_000

        for row in reader:
            row = row[:len(cols)]
            # normaliser longueur
            if len(row) < len(cols):
                row += [""] * (len(cols)-len(row))

            # null + cast types
            row2 = []
            for v, t in zip(row, types):
                v = normalize_null(v, null_tokens)
                row2.append(cast_for_type(v, t))
            batch.append(tuple(row2))
            nrows += 1

            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                batch.clear()

        if batch:
            cur.executemany(insert_sql, batch)

    # Index éventuels
    if create_indexes:
        for idx_col in create_indexes:
            sc = sanitize_col(idx_col)
            if sc not in cols:
                # si l'utilisateur a passé un nom "brut", tente correspondance loose
                # sinon, avertissement silencieux: on saute
                continue
            cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{sc}" ON "{table_name}" ("{sc}");')

    con.commit()
    return table_name, nrows

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Convertit CSV → SQLite (.db)")
    ap.add_argument("input_path", type=Path, help="Fichier CSV ou dossier contenant des CSV")
    ap.add_argument("sqlite_db", type=Path, help="Chemin de la base SQLite de sortie (*.db)")
    ap.add_argument("--table", type=str, default=None, help="Nom de table (si input est un seul fichier)")
    ap.add_argument("--if-exists", choices=["fail","replace","append"], default="replace",
                    help="Comportement si la table existe")
    ap.add_argument("--encoding", type=str, default="utf-8", help="Encodage des CSV")
    ap.add_argument("--delimiter", type=str, default=None, help="Forcer un délimiteur (ex: ';')")
    ap.add_argument("--nulls", type=str, default=",".join(sorted(NULL_TOKENS_DEFAULT)),
                    help="Liste de tokens NULL (séparés par des virgules)")
    ap.add_argument("--index", type=str, default=None,
                    help="Colonnes à indexer (séparées par des virgules)")
    ap.add_argument("--pk", type=str, default=None,
                    help="Nom de colonne à utiliser comme PRIMARY KEY (remplace/UPSERT sur collisions)")
    ap.add_argument("--infer-rows", type=int, default=5000,
                    help="Nombre de lignes pour inférer les types")
    args = ap.parse_args()

    null_tokens = {t.strip().lower() for t in args.nulls.split(",") if t.strip() != ""}
    index_cols = [c.strip() for c in args.index.split(",")] if args.index else None

    args.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.sqlite_db)

    try:
        if args.input_path.is_dir():
            csv_files = sorted([p for p in args.input_path.iterdir() if p.suffix.lower()==".csv"])
            if not csv_files:
                raise FileNotFoundError(f"Aucun CSV trouvé dans {args.input_path}")
            total = 0
            for csv_file in csv_files:
                table_name, n = import_csv_to_table(
                    csv_file, con,
                    table=None,
                    if_exists=args.if_exists,
                    encoding=args.encoding,
                    delimiter=args.delimiter,
                    null_tokens=null_tokens,
                    sample_rows_for_infer=args.infer_rows,
                    create_indexes=index_cols,
                    pk=args.pk,
                )
                print(f"[OK] {csv_file.name} → table '{table_name}' ({n} lignes)")
                total += n
            print(f"Terminé. Total lignes insérées: {total}")
        else:
            if args.table is None:
                # si fichier unique sans --table, déduit du nom de fichier
                args.table = sanitize_col(args.input_path.stem)
            table_name, n = import_csv_to_table(
                args.input_path, con,
                table=args.table,
                if_exists=args.if_exists,
                encoding=args.encoding,
                delimiter=args.delimiter,
                null_tokens=null_tokens,
                sample_rows_for_infer=args.infer_rows,
                create_indexes=index_cols,
                pk=args.pk,
            )
            print(f"[OK] {args.input_path.name} → table '{table_name}' ({n} lignes)")
            print("Terminé.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
