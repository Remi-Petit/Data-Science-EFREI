FROM python:3.11-slim

WORKDIR /app

COPY FastAPI/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY FastAPI/main.py ./main.py
COPY IA/Sujet_1/models ./IA/Sujet_1/models
COPY IA/Sujet_1/src ./IA/Sujet_1/src
COPY IA/Sujet_2/models ./IA/Sujet_2/models
COPY IA/Sujet_3/models ./IA/Sujet_3/models

ENV MODELS_DIR=/app/IA/Sujet_1/models
ENV S1_MODELS_DIR=/app/IA/Sujet_1/models
ENV S2_MODELS_DIR=/app/IA/Sujet_2/models
ENV S3_MODELS_DIR=/app/IA/Sujet_3/models
ENV PYTHONPATH=/app/IA/Sujet_1:$PYTHONPATH

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
