Project Path: Data

Source Tree:

```txt
Data
├── FastAPI
│   ├── __pycache__
│   │   └── main.cpython-311.pyc
│   └── main.py
├── IA
│   ├── industrial_machine_maintenance.csv
│   ├── logistic_regression_failure_24h.joblib
│   ├── maintenance_ml.ipynb
│   ├── random_forest_failure_24h.joblib
│   └── xgboost_failure_24h.joblib
├── README.md
└── Streamlit
    └── main.py

```

`FastAPI\main.py`:

```py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import List, Literal

app = FastAPI()

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]

# Chargement des modèles
MODELS = {
    "logistic_regression": joblib.load('../IA/logistic_regression_failure_24h.joblib'),
    "random_forest":       joblib.load('../IA/random_forest_failure_24h.joblib'),
    "xgboost":             joblib.load('../IA/xgboost_failure_24h.joblib'),
}

# Schéma des données d'entrée
class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    machine_type_enc: int
    operating_mode_enc: int
    hour: int
    dayofweek: int
    month: int
    models: List[ModelName] = ["random_forest"]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v):
        if not v:
            raise ValueError("La liste 'models' ne peut pas être vide.")
        return list(dict.fromkeys(v))  # dédoublonnage en conservant l'ordre

@app.get("/health")
def health():
    return {"status": "ok", "models_available": list(MODELS.keys())}

@app.post("/predict")
def predict(data: MachineData):
    features = data.model_dump(exclude={"models"})
    df = pd.DataFrame([features])

    results = {}
    for model_name in data.models:
        model = MODELS[model_name]
        prediction = int(model.predict(df)[0])
        probabilite = float(model.predict_proba(df)[0][1])
        results[model_name] = {
            "prediction": prediction,
            "label": "Panne probable" if prediction == 1 else "Pas de panne",
            "probabilite_panne": round(probabilite, 4)
        }

    return {"results": results}
```

`IA\industrial_machine_maintenance.csv`:

```csv
CSV Schema (1 sample row):
Headers: timestamp, machine_id, machine_type, vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm, operating_mode, hours_since_maintenance, ambient_temp, rul_hours, failure_within_24h, failure_type, estimated_repair_cost
Sample: "2024-01-01 00:00:00", "1", "CNC", "0.81", "49.51", "5.1", "23.6", "860.9", "idle", "273.8", "13.9", "61.0", "0", "none", "0"
... [24041 more rows omitted]

```

`IA\maintenance_ml.ipynb`:

```ipynb
Jupyter Notebook Summary:
Total cells: 30 (20 code, 10 markdown, 0 raw)

Code Cell #1:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid')
print('Librairies chargées ✓')
```

Code Cell #2:
```python
df = pd.read_csv('industrial_machine_maintenance.csv', parse_dates=['timestamp'])
print(f'Shape : {df.shape}')
df.head()
```

Code Cell #3:
```python
df.info()
```

... [17 more code cells omitted]

```

`README.md`:

```md
# Code2Prompt
code2prompt . --output-file architecture.md
```

`Streamlit\main.py`:

```py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}

st.title("🔧 Prédiction de panne machine")
st.markdown("Renseignez les paramètres de la machine pour prédire un risque de panne dans les 24h.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    vibration_rms = st.number_input("Vibration RMS", value=2.0, step=0.1)
    temperature_motor = st.number_input("Température moteur (°C)", value=70.0, step=0.5)
    current_phase_avg = st.number_input("Courant moyen (A)", value=10.0, step=0.1)
    ambient_temp = st.number_input("Température ambiante (°C)", value=25.0, step=0.5)

with col2:
    pressure_level = st.number_input("Pression (bar)", value=3.0, step=0.1)
    rpm = st.number_input("RPM", value=1450.0, step=10.0)
    hours_since_maintenance = st.number_input("Heures depuis maintenance", value=200.0, step=10.0)

with col3:
    machine_type_enc = st.selectbox("Type de machine", options=[0, 1, 2], format_func=lambda x: ["Pump", "Compressor", "Motor"][x])
    operating_mode_enc = st.selectbox("Mode opératoire", options=[0, 1, 2], format_func=lambda x: ["normal", "idle", "peak"][x])
    hour = st.slider("Heure", 0, 23, 12)
    dayofweek = st.slider("Jour de la semaine", 0, 6, 1)
    month = st.slider("Mois", 1, 12, 6)

st.divider()

selected_models = st.multiselect(
    "🤖 Modèles à comparer",
    options=list(MODEL_LABELS.keys()),
    default=["random_forest"],
    format_func=lambda x: MODEL_LABELS[x],
)

st.divider()

if st.button("🔍 Lancer la prédiction", use_container_width=True):
    if not selected_models:
        st.warning("Veuillez sélectionner au moins un modèle.")
        st.stop()

    payload = {
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp,
        "machine_type_enc": machine_type_enc,
        "operating_mode_enc": operating_mode_enc,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "models": selected_models,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        results = response.json()["results"]

        st.divider()
        st.subheader("📊 Résultats par modèle")

        cols = st.columns(len(results))
        for col, (model_key, res) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {MODEL_LABELS[model_key]}")
                if res["prediction"] == 1:
                    st.error(f"⚠️ {res['label']}")
                else:
                    st.success(f"✅ {res['label']}")
                st.metric("Probabilité de panne", f"{res['probabilite_panne'] * 100:.2f}%")

    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")

```