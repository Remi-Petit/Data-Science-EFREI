"""
Preprocessing pipeline – chargement, feature engineering, split train/test.
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'industrial_machine_maintenance.csv')

FEATURES = [
    'vibration_rms', 'temperature_motor', 'current_phase_avg',
    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp',
    'machine_type_enc', 'operating_mode_enc', 'hour', 'dayofweek', 'month'
]
TARGET = 'failure_within_24h'
TARGET_TYPE = 'failure_type'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut."""
    return pd.read_csv(path, parse_dates=['timestamp'])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features temporelles et encode les variables catégorielles."""
    df = df.copy()

    # Features temporelles
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Encodage label des variables catégorielles
    le_type = LabelEncoder()
    le_mode = LabelEncoder()
    df['machine_type_enc'] = le_type.fit_transform(df['machine_type'].astype(str))
    df['operating_mode_enc'] = le_mode.fit_transform(df['operating_mode'].astype(str))

    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Retourne X_train, X_test, y_train, y_test (split stratifié)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_type_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Retourne le split pour la classification du type de panne (multiclasses)."""
    X = df[FEATURES]
    y = df[TARGET_TYPE]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
