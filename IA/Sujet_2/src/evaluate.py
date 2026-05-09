"""
Évaluation des modèles de churn : métriques, matrices de confusion,
courbes ROC, importance des features, analyse du seuil de décision.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)


def evaluate_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcule Accuracy, Precision, Recall, F1, ROC-AUC et PR-AUC pour chaque modèle.
    Métriques centrées sur la classe Churn=1 (minoritaire).
    Retourne un DataFrame comparatif.
    """
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        report = classification_report(
            y_test, y_pred,
            target_names=['No Churn', 'Churn'],
            output_dict=True,
        )
        rows.append({
            'Modèle':    name,
            'Accuracy':  round(report['accuracy'], 4),
            'Precision': round(report['Churn']['precision'], 4),
            'Recall':    round(report['Churn']['recall'], 4),
            'F1-score':  round(report['Churn']['f1-score'], 4),
            'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
            'PR-AUC':    round(average_precision_score(y_test, y_prob), 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def plot_confusion_matrices(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Matrices de confusion côte à côte pour tous les modèles."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, trained_models.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    plt.suptitle('Matrices de confusion (jeu de test)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curves(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Courbes ROC de tous les modèles sur un même graphe."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    ax.set_title('Courbes ROC – Comparaison des modèles')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_pr_curves(trained_models: dict, X_test, y_test) -> plt.Figure:
    """
    Courbes Precision-Recall – plus informatives que ROC en cas de déséquilibre.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.set_title('Courbes Precision-Recall – Comparaison des modèles')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(
    trained_models: dict,
    features: list,
) -> dict:
    """
    Retourne un dict {model_name: Figure} avec un bar chart d'importance
    des features par modèle à arbres.
    """
    figures = {}
    for name, pipe in trained_models.items():
        if not hasattr(pipe.named_steps['clf'], 'feature_importances_'):
            continue
        importances = pipe.named_steps['clf'].feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        feat_imp.head(15).plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
        ax.set_title(f'Top 15 features – {name}')
        ax.set_ylabel('Importance')
        ax.tick_params(axis='x', rotation=35)
        plt.tight_layout()
        figures[name] = fig
    return figures


def plot_threshold_analysis(trained_models: dict, X_test, y_test) -> dict:
    """
    Retourne un dict {model_name: Figure} avec les courbes Precision / Recall / F1
    en fonction du seuil de décision, une figure par modèle.
    """
    from sklearn.metrics import precision_recall_curve

    figures = {}
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        denom = precisions[:-1] + recalls[:-1]
        f1_scores = np.where(
            denom > 0,
            2 * precisions[:-1] * recalls[:-1] / np.where(denom > 0, denom, 1),
            0,
        )
        best_idx = np.argmax(f1_scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(thresholds, precisions[:-1], label='Precision', color='steelblue')
        ax.plot(thresholds, recalls[:-1],   label='Recall',    color='tomato')
        ax.plot(thresholds, f1_scores,      label='F1-score',  color='green')
        ax.axvline(thresholds[best_idx], linestyle='--', color='gray',
                   label=f'Seuil optimal = {thresholds[best_idx]:.2f}')
        ax.set_title(f'Analyse du seuil – {name}')
        ax.set_xlabel('Seuil de décision')
        ax.legend()
        plt.tight_layout()
        figures[name] = fig
    return figures
