# merged.py
import argparse
from pathlib import Path
from functools import reduce
import pandas as pd

# ─── Ordre souhaité des fichiers (noms de fichiers attendus dans indir) ─── #


def main():
    ap = argparse.ArgumentParser(description="Fusion horizontale des CSV par email, dans un ordre imposé.")
    ap.add_argument("--year", type=int, required=True,
                    help="Année cible (ex: 2024). Utilisée pour construire les chemins par défaut.")
    ap.add_argument("--indir", type=Path, default=None,
                    help="Dossier contenant les CSV agrégés (défaut: data/data_fin)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Chemin du CSV fusionné en sortie (défaut: data/DATA.csv)")
    args = ap.parse_args()
    year_str = str(args.year)
    indir = args.indir or (Path("data") / "data_fin" / year_str)
    out_path = args.out or (Path("data") / f"DATA_{year_str}.csv")
    if not indir.exists():
        raise SystemExit(f"❌ Le dossier d'entrée n'existe pas: {indir.resolve()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.year == 2024:
        ORDERED_FILES = [ # 2024
        # C-Pool – Days 01 → 13
        "B-CPE-100_cpoolday01.csv",
        "B-CPE-100_cpoolday02.csv",
        "B-CPE-100_cpoolday03.csv",
        "B-CPE-100_cpoolday04.csv",
        "B-CPE-100_cpoolday05.csv",
        "B-CPE-100_cpoolday06.csv",
        "B-CPE-100_cpoolday07.csv",
        "B-CPE-100_cpoolday08.csv",
        "B-CPE-100_cpoolday09.csv",
        "B-CPE-100_cpoolday10.csv",
        "B-CPE-100_cpoolday11.csv",
        "B-CPE-100_cpoolday12.csv",
        "B-CPE-100_cpoolday13.csv",

        # B-CPE-110, B-PSU-100 et B-CPE-210 : rendus alternés
        "B-PSU-100_myls.csv",
        "B-CPE-110_settingup.csv",
        "B-PSU-100_mytop.csv",
        "B-CPE-110_organized.csv",
        "B-CPE-110_secured.csv",
        "B-PSU-100_mysudo.csv",

        # PSU-200, CPE-200 : rendus alternés
        "B-PSU-200_minishell1.csv",
        "B-CPE-200_robotfactory.csv",
        "B-PSU-200_minishell2.csv",
        "B-CPE-200_amazed.csv",
        "B-PSU-200_42sh.csv",
        # "B-CPE-200_corewar.csv"
    ]
    elif args.year == 2025:
        ORDERED_FILES = [ # 2025
        # C-Pool – Days 01 → 13
        "G-CPE-100_cpoolday01.csv",
        "G-CPE-100_cpoolday02.csv",
        "G-CPE-100_cpoolday03.csv",
        "G-CPE-100_cpoolday04.csv",
        "G-CPE-100_cpoolday05.csv",
        "G-CPE-100_cpoolday06.csv",
        "G-CPE-100_cpoolday07.csv",
        "G-CPE-100_cpoolday08.csv",
        "G-CPE-100_cpoolday09.csv",
        "G-CPE-100_cpoolday10.csv",
        "G-CPE-100_cpoolday11.csv",
        "G-CPE-100_cpoolday12.csv",
        "G-CPE-100_cpoolday13.csv"
        ]

    ordered_paths = [indir / f for f in ORDERED_FILES]
    missing = [p.name for p in ordered_paths if not p.exists()]
    if missing:
        print("⚠️  Fichiers manquants (ignorés) :", ", ".join(missing))
    existing_paths = [p for p in ordered_paths if p.exists()]
    if not existing_paths:
        print(f"❌ Aucun des fichiers attendus n'a été trouvé dans {indir.resolve()}")
        raise SystemExit(1)

    dfs = []
    for p in existing_paths:
        df = pd.read_csv(p)
        if "email" not in df.columns:
            raise ValueError(f"{p.name} ne contient pas de colonne 'email'")
        df = df.drop_duplicates(subset=["email", "city"])
        prefix = p.stem
        df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in ["email", "city"]})
        dfs.append(df)

    KEYS = ["email", "city"]
    merged = reduce(
    lambda left, right: pd.merge(left, right, on=KEYS, how="outer"),
    dfs
    )

    merged.to_csv(out_path, index=False)
    print(f"✅ CSV fusionné : {out_path}  –  {merged.shape[0]} lignes × {merged.shape[1]} colonnes")

if __name__ == "__main__":
    main()