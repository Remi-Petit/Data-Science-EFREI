"""
Évaluation des modèles : métriques, matrices de confusion, courbes ROC,
importance des features.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
)


def evaluate_models_type(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcule Accuracy, Precision (macro), Recall (macro), F1-macro et ROC-AUC (OvR macro)
    pour les modèles failure_type (multi-classes : none + types de panne).
    """
    classes = sorted(y_test.unique())
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        report = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0,
        )
        rows.append({
            'Modèle':     name,
            'Accuracy':   round(report['accuracy'], 4),
            'Precision':  round(report['macro avg']['precision'], 4),
            'Recall':     round(report['macro avg']['recall'], 4),
            'F1-score':   round(report['macro avg']['f1-score'], 4),
            'ROC-AUC':    round(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'), 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def evaluate_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcule accuracy, precision, recall, F1 et ROC-AUC pour chaque modèle.
    Retourne un DataFrame comparatif.
    """
    rows = []
    for name, pipe in trained_models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        report = classification_report(
            y_test, y_pred,
            target_names=['Pas de panne', 'Panne'],
            output_dict=True
        )
        rows.append({
            'Modèle':     name,
            'Accuracy':   round(report['accuracy'], 4),
            'Precision':  round(report['Panne']['precision'], 4),
            'Recall':     round(report['Panne']['recall'], 4),
            'F1-score':   round(report['Panne']['f1-score'], 4),
            'ROC-AUC':    round(roc_auc_score(y_test, y_prob), 4),
        })
    return pd.DataFrame(rows).set_index('Modèle')


def plot_confusion_matrices(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Affiche les matrices de confusion côte à côte."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, trained_models.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Pas de panne', 'Panne'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    plt.suptitle('Matrices de confusion (jeu de test)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curves(trained_models: dict, X_test, y_test) -> plt.Figure:
    """Courbes ROC pour tous les modèles sur un même graphe."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipe in trained_models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    ax.set_title('Courbes ROC – Comparaison des modèles')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(
    trained_models: dict,
    features: list,
) -> dict[str, plt.Figure]:
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
        fig, ax = plt.subplots(figsize=(9, 5))
        feat_imp.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
        ax.set_title(f'Importance des features – {name}')
        ax.set_ylabel('Importance')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        figures[name] = fig
    return figures
