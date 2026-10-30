# Mise en place rapide avec Docker
Une seule image : le backend Rust sert l'API (`/api`) et le frontend Svelte.

docker compose up -d --build

Aucun modèle ONNX n'est gardé en RAM : chacun est chargé pour la durée d'une requête puis libéré (~2 Mo au repos, ~45 Mo pendant une prédiction). `MODEL_IDLE_SECS=N` (N>0) les garde N secondes en cache, plus rapide mais plus gourmand.

## Éviter le problème de cache
docker compose build --no-cache

## Les URLS
App : http://localhost:40000
API : http://localhost:40000/api

# Mise en place en local
Il faut un env virtuel python de préférence

## Préparation du Backend Rust (Axum + ONNX Runtime)
Le backend ne lit que les `.onnx` (voir « Préparation des modèles d'IA » ci-dessous).

cd Backend && cargo run --release

## Préparation du Frontend Svelte
cd Frontend && npm install && npm run dev

Le serveur de dev proxifie `/api` vers `http://localhost:8000` (variable `API_TARGET`).

## Prépération des modèles d'IA
pip install -r IA/requirements.txt
python IA/train_all.py
python IA/export_onnx.py

## Les URLS
Frontend : http://localhost:5173
API : http://localhost:8000

# Code2Prompt
Un outil qui permet de combiner le code des différents fichiers du projet en un fichier markdown, facilement compréhensible par les LLM.

code2prompt . --output-file architecture.md