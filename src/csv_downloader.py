#!/usr/bin/env python3
"""
Microservice pour télécharger un fichier CSV depuis Google Drive
"""

import os
import argparse
import logging
import requests
import time
from pathlib import Path
from typing import Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoogleDriveDownloader:
    """Classe pour télécharger des fichiers depuis Google Drive"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_file_id(self, drive_url: str) -> str:
        """Extrait l'ID du fichier depuis une URL Google Drive"""
        if '/file/d/' in drive_url:
            file_id = drive_url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in drive_url:
            file_id = drive_url.split('id=')[1].split('&')[0]
        else:
            raise ValueError(f"Impossible d'extraire l'ID du fichier depuis l'URL: {drive_url}")
        
        logger.info(f"ID du fichier extrait: {file_id}")
        return file_id
    
    def download_file(self, file_id: str, output_path: str, max_retries: int = 3) -> bool:
        """
        Télécharge un fichier depuis Google Drive
        
        Args:
            file_id: ID du fichier Google Drive
            output_path: Chemin de destination pour sauvegarder le fichier
            max_retries: Nombre maximum de tentatives en cas d'échec
            
        Returns:
            bool: True si le téléchargement a réussi, False sinon
        """
        # URL pour le téléchargement direct
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Tentative {attempt + 1}/{max_retries} de téléchargement...")
                
                # Première requête pour obtenir le token de confirmation si nécessaire
                response = self.session.get(download_url, stream=True)
                response.raise_for_status()
                
                # Vérifier si Google demande une confirmation pour les gros fichiers
                if 'virus scan warning' in response.text.lower() or 'confirm=t' in response.text:
                    # Chercher le token de confirmation
                    confirm_token = None
                    for line in response.text.split('\n'):
                        if 'confirm=t' in line:
                            # Extraire le token depuis le HTML
                            parts = line.split('confirm=t&amp;')
                            if len(parts) > 1:
                                confirm_token = 't'
                                break
                    
                    if confirm_token:
                        download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                        response = self.session.get(download_url, stream=True)
                        response.raise_for_status()
                
                # Créer le répertoire de destination si nécessaire
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Télécharger le fichier
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                logger.info(f"Téléchargement vers: {output_path}")
                if total_size > 0:
                    logger.info(f"Taille du fichier: {total_size / (1024*1024):.2f} MB")
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Afficher le progrès pour les gros fichiers
                            if total_size > 0 and downloaded_size % (1024*1024) == 0:
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"Progrès: {progress:.1f}%")
                
                # Vérifier que le fichier a été téléchargé correctement
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    final_size = os.path.getsize(output_path)
                    logger.info(f"Téléchargement réussi! Taille finale: {final_size / (1024*1024):.2f} MB")
                    return True
                else:
                    logger.error("Le fichier téléchargé est vide ou n'existe pas")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur de réseau lors de la tentative {attempt + 1}: {e}")
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la tentative {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Backoff exponentiel
                logger.info(f"Attente de {wait_time} secondes avant la prochaine tentative...")
                time.sleep(wait_time)
        
        logger.error(f"Échec du téléchargement après {max_retries} tentatives")
        return False
    
    def download_from_url(self, drive_url: str, output_path: str, max_retries: int = 3) -> bool:
        """
        Télécharge un fichier depuis une URL Google Drive complète
        
        Args:
            drive_url: URL complète du fichier Google Drive
            output_path: Chemin de destination pour sauvegarder le fichier
            max_retries: Nombre maximum de tentatives en cas d'échec
            
        Returns:
            bool: True si le téléchargement a réussi, False sinon
        """
        try:
            file_id = self.extract_file_id(drive_url)
            return self.download_file(file_id, output_path, max_retries)
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement depuis l'URL {drive_url}: {e}")
            return False


def main():
    """Fonction principale du microservice"""
    parser = argparse.ArgumentParser(description='Télécharger un fichier CSV depuis Google Drive')
    parser.add_argument('--url', required=True, help='URL du fichier Google Drive')
    parser.add_argument('--output', required=True, help='Chemin de sortie pour le fichier CSV')
    parser.add_argument('--retries', type=int, default=3, help='Nombre maximum de tentatives (défaut: 3)')
    parser.add_argument('--verify', action='store_true', help='Vérifier que le fichier est un CSV valide')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("MICROSERVICE DE TÉLÉCHARGEMENT CSV")
    logger.info("=" * 50)
    logger.info(f"URL source: {args.url}")
    logger.info(f"Destination: {args.output}")
    logger.info(f"Tentatives max: {args.retries}")
    
    # Créer le téléchargeur
    downloader = GoogleDriveDownloader()
    
    # Télécharger le fichier
    success = downloader.download_from_url(args.url, args.output, args.retries)
    
    if success:
        logger.info("✅ Téléchargement terminé avec succès!")
        
        # Vérification optionnelle du format CSV
        if args.verify:
            try:
                import pandas as pd
                df = pd.read_csv(args.output, nrows=5)  # Lire seulement les 5 premières lignes
                logger.info(f"✅ Fichier CSV valide détecté. Colonnes: {list(df.columns)}")
                logger.info(f"✅ Nombre de colonnes: {len(df.columns)}")
            except Exception as e:
                logger.warning(f"⚠️  Impossible de vérifier le format CSV: {e}")
        
        # Afficher les informations du fichier
        file_size = os.path.getsize(args.output)
        logger.info(f"📊 Taille du fichier: {file_size / (1024*1024):.2f} MB")
        
        exit(0)
    else:
        logger.error("❌ Échec du téléchargement!")
        exit(1)


if __name__ == "__main__":
    main()