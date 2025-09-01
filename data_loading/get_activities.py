#!/usr/bin/env python3
"""
Télécharge *toutes* les pages d’/api/activities (year=2024, intra, B-CPE-100)
et les enregistre dans test2/activities_2024_B-CPE-100.json.

⚠️  Pas de pause → si l’API atteint son quota, un HTTPError 429 stoppera le script.
"""
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# ─────────── paramètres fixes ─────────── #
BASE_URL = "https://api.epitest.eu/api"
FILTERS  = {"year": 2024, "source": "intra", "unit": "B-CPE-110"}
EXPORTS  = Path("test2"); EXPORTS.mkdir(exist_ok=True)
# ──────────────────────────────────────── #

load_dotenv(override=True)
TOKEN_ID, TOKEN_PWD = os.environ["TOKEN_ID"], os.environ["TOKEN_PASS"]

def fetch_page(page: int) -> dict:
    """Appelle /activities?page=N. Lève HTTPError si statut ≠ 2xx."""
    url    = f"{BASE_URL}/activities"
    params = FILTERS | {"page": page}
    resp   = requests.get(url, auth=HTTPBasicAuth(TOKEN_ID, TOKEN_PWD),
                          params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def main() -> None:
    all_activities, page = [], 1

    while True:
        payload = fetch_page(page)
        chunk   = payload.get("activities", [])
        all_activities.extend(chunk)

        print(f"✓ page {page} : {len(chunk)} act. – total {len(all_activities)}")

        if not payload.get("hasNextPage"):
            break
        page += 1   # enchaîne immédiatement la page suivante (pas de pause)

    out_file = EXPORTS / "activities_2024_B-CPE-100.json"
    out_file.write_text(
        json.dumps({"activities": all_activities}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✅  {len(all_activities)} activité(s) sauvegardées dans {out_file}")

if __name__ == "__main__":
    main()
