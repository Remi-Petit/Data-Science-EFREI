FROM python:3.11-slim

WORKDIR /app

COPY FastAPI/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY FastAPI/main.py ./main.py
COPY IA/Sujet_1/models ./models

ENV MODELS_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
