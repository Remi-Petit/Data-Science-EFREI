# Mise en place rapide avec Docker
docker compose up -d --build

## Éviter le problème de cache
docker compose build --no-cache

## Les URLS
Frontend : http://localhost:40000
API : http://localhost:40001

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