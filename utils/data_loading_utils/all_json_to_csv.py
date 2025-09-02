#!/usr/bin/env python3
"""
Pour chaque test_results_*.json dans un arborescence récursive sous un dossier d'entrée,
crée le CSV correspondant en conservant la structure des sous-dossiers
sous le dossier de sortie.

Ex. :
    python json2csv_batch.py real_data_json/ real_data_csv/
    -> transforme real_data_json/2023/foo/*.json en real_data_csv/2023/foo/*.csv
    python json2csv_batch.py fichier.json out/        
    -> CSV unique dans out/fichier.csv
"""
import json, csv, argparse
from pathlib import Path

# ───────────────────────── helpers « 1 fichier » ───────────────────────── #

def list_tasks(results):
    return sorted({t for r in results for t in r.get("skillBreakdowns", {})})

def passed(entry):
    return entry[0].get("passed", 0) if entry else 0

def style_fields(result):
    s = (result.get("style", {})
                  .get("CodingStyleSummary", {})
                  .get("summary", {}))
    c = s.get("value", {})
    return {
        "stylePenalty": s.get("penalty", {}).get("Penalty", {}).get("value"),
        "styleFatal":   c.get("fatal"),
        "styleMajor":   c.get("major"),
        "styleMinor":   c.get("minor"),
        "styleInfo":    c.get("info"),
    }

def rows_for_file(data, tasks):
    rows = []
    for r in data.get("results", []):
        members = r.get("properties", {}).get("group", {}).get("members", [])
        sb      = r.get("skillBreakdowns", {})
        per_task = {t: passed(sb.get(t)) for t in tasks}

        for email in members:
            row = {
                "email": email,
                "mark": r.get("mark"),
                "virtualMark": r.get("virtualMark"),
                "prerequisitesMark": r.get("prerequisitesMark"),
                **style_fields(r),
                **{f"{t}_passed": per_task[t] for t in tasks},
            }
            rows.append(row)
    return rows

def convert_one(json_path: Path, csv_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    tasks = list_tasks(data.get("results", []))
    if not tasks:
        print(f"⚠️  {json_path} : pas de résultats.")
        return

    rows = rows_for_file(data, tasks)
    fieldnames = (
        ["email", *[f"{t}_passed" for t in tasks],
         "mark", "virtualMark", "prerequisitesMark",
         "stylePenalty", "styleFatal", "styleMajor", "styleMinor", "styleInfo"]
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅  {csv_path} ({len(rows)} lignes)")

# ──────────────────────────── programme ──────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="JSON → CSV récursif")
    ap.add_argument(
        "--input", type=Path,
                    help="Fichier JSON ou dossier racine contenant des .json"
                    )
    ap.add_argument(
        "--output", type=Path, default=Path("data_csv"),
        help="Dossier de sortie (défaut: real_data_csv)"
                    )
    args = ap.parse_args()

    if args.input.is_dir():
        base_input = args.input
        base_output = args.output
        json_files = sorted(base_input.rglob("*.json"))
    else:
        base_input = args.input.parent
        base_output = args.output
        json_files = [args.input]

    if not json_files:
        print("⚠️  Aucun .json trouvé.")
        return

    for jf in json_files:
        rel = jf.relative_to(base_input)
        csv_path = base_output / rel.with_suffix('.csv')
        convert_one(jf, csv_path)

if __name__ == "__main__":
    main()
