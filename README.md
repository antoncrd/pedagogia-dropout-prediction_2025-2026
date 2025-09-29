# Pedagogia Dropout Prediction · 2025–2026

Système modulaire pour **prévoir le risque de décrochage** à partir des traces pédagogiques (projets, tests unitaires, notes). 
Le pipeline couvre : téléchargement automatique des données depuis Google Drive, ingestion des données, normalisation en CSV, agrégation/merge par étudiant, puis **production d'un modèle** (clustering + prédiction séquentielle) avec métriques.

> Tech: Python 3.12, scikit‑learn, UMAP, Docker (optionnel).gogia Dropout Prediction · 2025–2026

Système modulaire pour **prévoir le risque de décrochage** à partir des traces pédagogiques (projets, tests unitaires, notes). 
Le pipeline couvre : ingestion des données, normalisation en CSV, agrégation/merge par étudiant, puis **production d’un modèle** (clustering + prédiction séquentielle) avec métriques.

> Tech: Python 3.12, scikit‑learn, UMAP, Docker (optionnel).

---

## ✨ Points clés

- **Ingestion** depuis les APIs internes (pagination, back‑off 429, export JSON par activité).
- **Normalisation CSV** + **agrégation** (verticale par préfixe) + **fusion horizontale** par `email`.
- **Production modèle** (script principal) avec paramètres explicites : `year`, `n_clusters`, `min_cluster_size`, seuils, fenêtres `w1/w2`, alphas.
- **Architecture simple** : tout le code est sous `src/` avec utilitaires dédiés dans `src/utils/…`.
- **Docker prêt à l’emploi** (Dockerfile + docker‑compose.yml).

---

## 📁 Structure du dépôt

```
pedagogia-dropout-prediction_2025-2026/
├─ src/
│  ├─ data_loading.py                         # Orchestration pipeline (ingestion → CSV → agrégat → merge)
│  ├─ model_production_main.py                # Script principal de production du modèle
│  └─ utils/
│     ├─ data_loading_utils/
│     │  ├─ get_data.py                       # Télécharge JSON (résultats « delivery ») pour une unité/année
│     │  ├─ all_json_to_csv.py                # Conversion récursive JSON → CSV
│     │  ├─ aggregate_csv.py                  # Agrégation verticale par « préfixe »
│     │  ├─ merged_csv.py                     # Fusion horizontale par email (ordre imposé si besoin)
│     │  └─ get_activities.py                 # Utilitaire activités (pagination)
│     └─ model_production_data_processing_utils.py
├─ tests.ipynb
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

## 🧰 Prérequis

- **Python 3.12+**
- (Optionnel) **Docker** ≥ 24
- Accès API + **variables d’environnement** (via `.env`) :  
  - `TOKEN_ID`  
  - `TOKEN_PASS`

Créez un fichier `.env` à la racine :

```env
TOKEN_ID=xxxxxxxxxxxxxxxx
TOKEN_PASS=yyyyyyyyyyyyyyyy
```

> Les scripts chargent `.env` via `dotenv` et utilisent ces tokens pour l’appel API.

---

## � Téléchargement automatique des données

Le système inclut un **microservice de téléchargement** qui récupère automatiquement le fichier CSV depuis Google Drive.


**via Docker Compose:**
```bash
docker compose up csv-downloader --build
```

Le fichier sera téléchargé dans `./data/DATA_2025_pred_proj.csv`.

### Configuration manuelle

Vous pouvez aussi utiliser le microservice directement :

```bash
python src/csv_downloader.py \
  --url "https://drive.google.com/file/d/1ZeJ2f1qfpENc-gIwYGiQsVM5nCGaTQkG/view?usp=drive_link" \
  --output "./data/DATA_2025_pred_proj.csv" \
  --retries 3 \
  --verify
```

---

## �🚀 Installation rapide (sans Docker)

```bash
# 1) Cloner et se placer dans le dossier
git clone <votre-repo> pedagogia-dropout-prediction_2025-2026
cd pedagogia-dropout-prediction_2025-2026

# 2) Créer un venv et installer
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🔄 Pipeline de données (ingestion → CSV → agrégat → merge)

Vous pouvez soit appeler les sous‑scripts, soit piloter via `src/data_loading.py`. Ci‑dessous, la version **pas à pas** (recommandé pour être explicite sur les chemins).

### 1) Télécharger les JSON (*results « delivery »*)

Script : `src/utils/data_loading_utils/get_data.py`

Arguments détectés : `--year` (int, requis), `--unit` (str, requis), `--source` (intra/extra, défaut *intra*), `--output` (dossier de sortie).

```bash
python -u src/utils/data_loading_utils/get_data.py   --year 2024   --unit B-CPE-100   --source intra   --output data/data_json
```

> ❗️Si vous voyez l’erreur `the following arguments are required: --year, --unit`, passez bien ces 2 paramètres (ils sont **requis**).

### 2) Convertir JSON → CSV

```bash
python -u src/utils/data_loading_utils/all_json_to_csv.py   --input data/data_json   --output data_csv
```

### 3) Agréger verticalement (par « préfixe » de fichier)

Le préfixe est tout avant le **dernier** underscore `_`. Tous les fichiers partageant ce préfixe sont concaténés.

```bash
python -u src/utils/data_loading_utils/aggregate_csv.py   --indir data_csv   --outdir agg   --filter "*.csv"
```

### 4) Fusion horizontale par `email`

```bash
python -u src/utils/data_loading_utils/merged_csv.py   --indir agg   --out data/DATA.csv
```

> À l’issue de ces 4 étapes, vous disposez d’un **master CSV** : `data/DATA.csv`

---

## 🤖 Production du modèle

Script : `src/model_production_main.py`  
Arguments disponibles : `--year`, `--n_clusters`, `--min_cluster_size`, `--data_file`, `--threshold`, `--w1`, `--w2`, `--alpha1`, `--alpha2`.

Exemple minimal reproductible :

```bash
python -u src/model_production_main.py   --year 2024   --n_clusters 10   --min_cluster_size 20   --data_file data/DATA.csv   --threshold 10.0   --w1 2 --w2 2   --alpha1 0.05 --alpha2 0.05
```

> Notes :
> - `n_clusters` et `min_cluster_size` contrôlent le **clustering** amont.
> - `w1/w2` et `alpha1/alpha2` contrôlent la **prédiction séquentielle** (p. ex. variantes SPCI) et les budgets d’alpha.
> - `threshold` peut servir de seuil métier (ex. décision d’alerte).

---

## 🧪 Reproductibilité

- Versions **pinnées** dans `requirements.txt`.
- Fixez vos graines globales (ex. `RANDOM_STATE=42`) au niveau des scripts si besoin.
- Exportez vos **artefacts** (modèles, métriques) dans un dossier versionné (`models/`, `reports/`).

---

## 🐳 Exécution via Docker (optionnel)

### Build image

```bash
docker build -t pedagogia-dropout:latest .
```

### Lancer le pipeline dans un conteneur

```bash
docker run --rm -it   --env-file .env   -v "$(pwd)/data:/app/data"   -v "$(pwd)/models:/app/models"   pedagogia-dropout:latest   python -u src/utils/data_loading_utils/get_data.py --year 2024 --unit B-CPE-100 --output data/data_json
```

> Vous pouvez ensuite enchaîner les étapes 2→4, puis lancer `model_production_main.py` de la même façon.

> **docker-compose** : un fichier `docker-compose.yml` est fourni ; adaptez les volumes `data/` et `models/` selon votre environnement.

---

## 🗂️ Données attendues (résumé)

- **Entrées** : JSON par activité \→ CSV normalisés \→ agrégats par préfixe \→ merge final par `email`.
- **Sortie** : `data/DATA.csv` (master), + artefacts de modèles dans `models/` (si configuré).
