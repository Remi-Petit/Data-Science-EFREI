"""
Preprocessing pipeline – chargement, feature engineering, split train/test.
Dataset : marketing_and_sales.csv (~4572 campagnes marketing)
Tâche principale : Régression (prédiction des ventes)
Tâche bonus     : Classification (performance campagne : Low / Medium / High)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'marketing_and_sales.csv',
)

# ── Encodage Influencer (ordre alphabétique = LabelEncoder sklearn) ───────────
INFLUENCER_MAP = {'Macro': 0, 'Mega': 1, 'Micro': 2, 'Nano': 3}

# ── Seuils de classification (quantiles 33 % / 66 %) ─────────────────────────
PERF_LOW_THRESHOLD  = 136.86   # 33e percentile
PERF_HIGH_THRESHOLD = 241.53   # 66e percentile
PERF_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# ── Features utilisées par les modèles ───────────────────────────────────────
FEATURES = [
    'tv', 'radio', 'social_media', 'influencer_enc',
    'total_budget', 'tv_share', 'radio_share', 'social_share',
    'tv_social_interaction',
]

TARGET_REG   = 'sales'
TARGET_CLASS = 'performance'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et enrichit le DataFrame :
    - Renommage des colonnes
    - Suppression des lignes sans cible
    - Imputation médiane des variables numériques
    - Encodage Influencer
    - Feature engineering métier
    - Variable cible classification (Low / Medium / High)
    """
    df = df.copy()

    # ── Renommage ─────────────────────────────────────────────────────────────
    df = df.rename(columns={
        'TV':           'tv',
        'Radio':        'radio',
        'Social Media': 'social_media',
        'Influencer':   'influencer',
        'Sales':        'sales',
    })

    # ── Suppression des lignes sans cible ─────────────────────────────────────
    df = df.dropna(subset=['sales'])

    # ── Imputation médiane (variables numériques) ─────────────────────────────
    for col in ['tv', 'radio', 'social_media']:
        df[col] = df[col].fillna(df[col].median())

    # ── Encodage Influencer ───────────────────────────────────────────────────
    df['influencer_enc'] = df['influencer'].map(INFLUENCER_MAP)

    # ── Feature engineering métier ───────────────────────────────────────────
    df['total_budget']          = df['tv'] + df['radio'] + df['social_media']
    df['tv_share']              = df['tv']           / df['total_budget']
    df['radio_share']           = df['radio']        / df['total_budget']
    df['social_share']          = df['social_media'] / df['total_budget']
    df['tv_social_interaction'] = df['tv'] * df['social_media']

    # ── Variable cible classification (entier : 0=Low, 1=Medium, 2=High) ─────
    bins = [-np.inf, PERF_LOW_THRESHOLD, PERF_HIGH_THRESHOLD, np.inf]
    df['performance'] = pd.cut(
        df['sales'], bins=bins, labels=[0, 1, 2],
    ).astype(int)

    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split stratifié sur la performance.
    Retourne : X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
    """
    X      = df[FEATURES]
    y_reg  = df[TARGET_REG]
    y_cls  = df[TARGET_CLASS]

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cls,
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
