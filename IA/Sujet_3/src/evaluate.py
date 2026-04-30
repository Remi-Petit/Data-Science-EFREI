"""
Évaluation des modèles de régression et classification marketing.
Métriques régression    : MAE, RMSE, R²
Métriques classification : Accuracy, F1-macro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def evaluate_regression(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Calcule MAE, RMSE et R² pour chaque modèle de régression."""
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        rows.append({
            'Modèle': name,
            'MAE':    round(mean_absolute_error(y_test, y_pred), 4),
            'RMSE':   round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'R²':     round(r2_score(y_test, y_pred), 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def evaluate_classification(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Calcule Accuracy et F1-macro pour chaque modèle de classification."""
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=['Low', 'Medium', 'High'],
            output_dict=True,
            zero_division=0,
        )
        rows.append({
            'Modèle':     name,
            'Accuracy':   round(report['accuracy'], 4),
            'F1 (macro)': round(report['macro avg']['f1-score'], 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def get_feature_importance(pipe, feature_names: list) -> pd.DataFrame:
    """Extrait l'importance des variables pour les modèles tree-based."""
    estimator = pipe.named_steps.get('reg') or pipe.named_steps.get('clf')
    if not hasattr(estimator, 'feature_importances_'):
        return pd.DataFrame()
    return (
        pd.DataFrame({'Feature': feature_names, 'Importance': estimator.feature_importances_})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )


def plot_predictions_vs_actual(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Scatter plot prédit vs réel pour chaque modèle de régression."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, trained_models.items()):
        y_pred = pipe.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.4, s=15)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel('Ventes réelles')
        ax.set_ylabel('Ventes prédites')
        ax.set_title(name)
    plt.suptitle('Prédit vs Réel – Régression des ventes', fontsize=14)
    plt.tight_layout()
    return fig
