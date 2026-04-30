"""
Preprocessing pipeline – chargement, feature engineering, split train/test.
Dataset : customer_churn.csv (10 000 clients, cible : churn binaire)
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'customer_churn.csv'
)

# Variables numériques brutes
NUM_FEATURES = [
    'age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
    'avg_session_time', 'features_used', 'usage_growth_rate',
    'last_login_days_ago', 'monthly_fee', 'total_revenue',
    'payment_failures', 'avg_resolution_time', 'csat_score',
    'escalations', 'email_open_rate', 'marketing_click_rate',
    'nps_score', 'referral_count',
]

# Variables catégorielles à encoder
CAT_FEATURES = [
    'gender', 'customer_segment', 'signup_channel',
    'contract_type', 'payment_method', 'discount_applied',
    'price_increase_last_3m', 'survey_response', 'complaint_type',
]

# Features engineered (ajoutées dans engineer_features)
ENGINEERED_FEATURES = [
    'revenue_per_month',
    'engagement_score',
    'ticket_burden',
]

# Liste complète des features utilisées pour l'entraînement
FEATURES = NUM_FEATURES + [f + '_enc' for f in CAT_FEATURES] + ENGINEERED_FEATURES

TARGET = 'churn'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et enrichit le DataFrame :
    - Imputation de complaint_type (valeur manquante → 'Unknown')
    - Encodage label des variables catégorielles
    - Feature engineering métier
    """
    df = df.copy()

    # ── Imputation ──────────────────────────────────────────────────────────
    df['complaint_type'] = df['complaint_type'].fillna('Unknown')

    # ── Encodage des variables catégorielles ─────────────────────────────────
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    # ── Feature engineering métier ───────────────────────────────────────────
    # Revenu mensuel moyen sur la durée du contrat
    df['revenue_per_month'] = df['total_revenue'] / (df['tenure_months'] + 1)

    # Score d'engagement : fréquence × durée de session
    df['engagement_score'] = df['weekly_active_days'] * df['avg_session_time']

    # Charge de support pondérée : tickets × temps de résolution moyen
    df['ticket_burden'] = df['support_tickets'] * (df['avg_resolution_time'] + 1)

    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Retourne X_train, X_test, y_train, y_test (split stratifié sur churn)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
