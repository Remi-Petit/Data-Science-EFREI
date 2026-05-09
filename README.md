# Mise en place rapide avec Docker
docker compose up -d --build

## Éviter le problème de cache
docker compose build --no-cache

## Les URLS

Streamlit : http://localhost:8501
API : http://localhost:8000
Doc : http://localhost:8000/docs

# Mise en place en local
Il faut un env virtuel python de préférence

## Préparation du Backend FastAPI
pip install -r FastAPI/requirements.txt
fastapi dev main.py

## Préparation du Frontend Streamlit
pip install -r Streamlit/requirements.txt
streamlit run main.py

## Prepération des modèles d'IA
pip install -r IA/requirements.txt
python IA/train_all.py

# Code2Prompt
Un outil qui permet de combiner le code des différents fichiers du projet en un fichier markdown, facilement compréhensible par les LLM.

code2prompt . --output-file architecture.md