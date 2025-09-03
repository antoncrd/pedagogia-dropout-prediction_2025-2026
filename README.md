# Pedagogia Dropout Prediction 2025-2026

Un système de prédiction de décrochage scolaire utilisant l'analyse de données pédagogiques et des techniques de machine learning.

## 🚀 Fonctionnalités

- **Chargement et traitement des données** : Pipeline automatisé pour récupérer, convertir et agréger les données pédagogiques
- **Analyse prédictive** : Modèles de machine learning pour prédire le risque de décrochage
- **Clustering avancé** : Utilisation d'algorithmes de clustering avec gestion des tailles minimales
- **Visualisation** : Graphiques et analyses pour interpréter les résultats
- **Architecture modulaire** : Services séparés pour le chargement des données et la production de modèles

## 📋 Prérequis

- Python 3.11+
- Docker (optionnel, recommandé)
- Git

## 🛠️ Installation

### Avec Docker (Recommandé)

1. **Cloner le repository** :
   ```bash
   git clone https://github.com/antoncrd/pedagogia-dropout-prediction_2025-2026.git
   cd pedagogia-dropout-prediction_2025-2026
   ```

2. **Construire et lancer les services** :
   ```bash
   docker-compose up --build
   ```

### Sans Docker

1. **Cloner le repository** :
   ```bash
   git clone https://github.com/antoncrd/pedagogia-dropout-prediction_2025-2026.git
   cd pedagogia-dropout-prediction_2025-2026
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Utilisation

### Avec Docker Compose

```bash
# Lancer tous les services
docker-compose up

# Lancer en arrière-plan
docker-compose up -d

# Suivre les logs
docker-compose logs -f

# Arrêter les services
docker-compose down
```

### Scripts individuels

#### Chargement des données
```bash
python -u data_loading_julien.py --year 2024 --data_dir data --utils_dir utils
```

#### Production de modèles
```bash
python -u model_production_main_julien.py --year 24 --n_clusters 4 --data_file data/DATA.csv
```

### Arguments disponibles

#### data_loading_julien.py
- `--year` : Année académique (défaut: 2024)
- `--data_dir` : Dossier des données (défaut: data)
- `--utils_dir` : Dossier des utilitaires (défaut: utils)

#### model_production_main_julien.py
- `--year` : Année académique (défaut: 24)
- `--n_clusters` : Nombre de clusters (défaut: 4)
- `--min_cluster_size` : Taille minimale des clusters (défaut: 50)
- `--data_file` : Fichier de données CSV (défaut: data/DATA.csv)
- `--threshold` : Seuil d'analyse (défaut: 0.5)

## 📁 Structure du projet

```
pedagogia-dropout-prediction_2025-2026/
├── data/                    # Données d'entrée et de sortie
├── utils/                   # Utilitaires et scripts
│   ├── data_loading_utils/  # Scripts de chargement des données
│   └── models_production_utils.py  # Utilitaires pour les modèles
├── src/                     # Code source partagé
├── models/                  # Modèles sauvegardés
├── data_loading_julien.py   # Script de chargement des données (amélioré)
├── model_production_main_julien.py  # Script principal de production (amélioré)
├── requirements.txt          # Dépendances Python
├── Dockerfile               # Configuration Docker
├── docker-compose.yml       # Orchestration des services
└── README.md               # Ce fichier
```
