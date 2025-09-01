#!/usr/bin/env python3
"""
Télécharge tous les résultats « delivery » des projets B-CPE-100 (2024, intra).

• Pagination page/hasNextPage côté /activities             (50 act. par page)
• Gestion du quota : back-off exponentiel sur 429 Too Many Requests
• Un JSON par activité : test/<code>_<slug>_<instance>.json
"""
import os
import re
import json
import time
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import email.utils
import datetime
import argparse

# ───────────────────────── CONFIG ───────────────────────── #
BASE_URL     = "https://api.epitest.eu/api"
RUN_TYPE     = "delivery"
EXPORTS      = Path("real_data_json")
EXPORTS.mkdir(exist_ok=True)

PAGE_PAUSE   = 0            # s entre pages /activities (0 = aucune pause)
RATE_DELAY   = 0.5          # s après chaque test_results réussi
MAX_RETRIES  = 10           # tentatives max par test_results
BASE_BACKOFF = 2            # s avant 1er retry, doublé ensuite (max 60 s)
# ─────────────────────────────────────────────────────────── #

load_dotenv(override=True)
TOKEN_ID  = os.environ["TOKEN_ID"]
TOKEN_PWD = os.environ["TOKEN_PASS"]

SESSION   = requests.Session()  # réutilise la connexion TCP

def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge les résultats de tests 'delivery' d'activités Epitest."
    )
    parser.add_argument("--year",   type=int,   required=True, help="Année (ex: 2023)")
    parser.add_argument("--source", type=str,   default="intra", choices=["intra","extra"], help="Source (intra/extrasite/... selon API)")
    parser.add_argument("--unit",   type=str,   required=True, help="Code de l'unité (ex: B-CPE-100)")
    parser.add_argument("--output", type=str,   default="data/data_json", help="Dossier de sortie")
    return parser.parse_args()

def sanitize(text: str) -> str:
    """Remplace tout caractère exotique pour un nom de fichier sûr."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)) or "unknown"

def fetch_all_activities(filters: dict) -> list[dict]:
    activities, page = [], 1
    while True:
        resp = SESSION.get(
            f"{BASE_URL}/activities",
            auth=HTTPBasicAuth(TOKEN_ID, TOKEN_PWD),
            params={**filters, "page": page},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        chunk   = payload.get("activities", [])
        activities.extend(chunk)

        print(f"✓ page {page}: {len(chunk)} act.  (total {len(activities)})")
        if not payload.get("hasNextPage"):
            break
        page += 1
        if PAGE_PAUSE:
            time.sleep(PAGE_PAUSE)
    return activities

def seconds_until(date_http: str) -> float:
    ts = email.utils.parsedate_to_datetime(date_http)
    now = datetime.datetime.now(datetime.timezone.utc)
    return max((ts - now).total_seconds(), 0)

def get_test_results(activity_id: int) -> dict:
    url = f"{BASE_URL}/activities/i/{activity_id}/test_results/{RUN_TYPE}"
    delay = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        r = SESSION.get(url, auth=HTTPBasicAuth(TOKEN_ID, TOKEN_PWD), timeout=15)
        if r.status_code != 429:
            r.raise_for_status()
            if RATE_DELAY:
                time.sleep(RATE_DELAY)
            return r.json()

        retry_hdr = r.headers.get("Retry-After")
        if retry_hdr is None:
            wait = delay
        else:
            try:
                wait = int(retry_hdr)
            except ValueError:
                wait = seconds_until(retry_hdr)
        wait = max(wait, delay)
        print(f"  429 (id {activity_id}) essai {attempt}/{MAX_RETRIES} → attente {wait:.1f}s", file=sys.stderr)
        time.sleep(wait)
        delay = min(delay * 2, 60)

    raise RuntimeError(f"Abandon : {MAX_RETRIES} échecs 429 pour l’activité {activity_id}")

def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[ok] {path.name}")

def main():
    args = parse_args()
    # Préparer le dossier de sortie
    EXPORT_DIR = Path(args.output) / str(args.year) / args.unit
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Filtres dynamiques
    filters = {
        "year":   args.year,
        "source": args.source,
        "unit":   args.unit,
    }

    acts = fetch_all_activities(filters)
    if not acts:
        print("Aucune activité trouvée ; vérifiez filtres ou VPN.")
        return

    print(f"\nTéléchargement des tests (total {len(acts)}) …\n")
    for act in acts:
        act_id   = act["id"]
        code     = sanitize(act.get("unitTemplate", {}).get("code"))
        slug     = sanitize(act.get("projectTemplate", {}).get("slug"))
        instance = sanitize(act.get("instance", {}).get("instanceCode"))

        try:
            payload = get_test_results(act_id)
        except Exception as e:
            print(f"[err] id {act_id} ({slug}) : {e}", file=sys.stderr)
            continue

        filename = f"{code}_{slug}_{instance}.json"
        save_json(payload, EXPORT_DIR / filename)

    total_files = len(list(EXPORT_DIR.glob("*.json")))
    print(f"\n✅ Terminé : {total_files} fichiers dans {EXPORT_DIR}/")

if __name__ == "__main__":
    main()
