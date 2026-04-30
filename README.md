# Mise en place rapide avec Docker
docker compose up -d --build

## Éviter le problème de cache
docker compose build --no-cache

# Les URLS

Streamlit : http://localhost:8501
API : http://localhost:8000
Doc : http://localhost:8000/docs

# Code2Prompt
Un outil qui permet de combiner le code des différents fichiers du projet en un fichier markdown, facilement compréhensible par les LLM.

code2prompt . --output-file architecture.md