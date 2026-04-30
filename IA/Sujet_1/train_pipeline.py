"""
Pipeline d'entraînement – exécutable en ligne de commande.

Usage (depuis IA/Sujet_1/) :
    python train_pipeline.py
"""
import os
import sys

# Résolution des imports locaux quand on exécute le script directement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_data, engineer_features, get_train_test_split, FEATURES
from src.train import build_pipelines, cross_validate_models, train_and_save
from src.evaluate import evaluate_models, plot_confusion_matrices, plot_roc_curves, plot_feature_importance


def main():
    # ── 1. Données ────────────────────────────────────────────────────────────
    print("=== Chargement et préparation des données ===")
    df = load_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"Train : {X_train.shape[0]} lignes  |  Test : {X_test.shape[0]} lignes")
    print(f"Taux de pannes (train) : {y_train.mean():.2%}")

    # ── 2. Cross-validation ───────────────────────────────────────────────────
    print("\n=== Cross-validation (StratifiedKFold 5 folds) ===")
    models = build_pipelines()
    cv_results = cross_validate_models(models, X_train, y_train)
    for name, scores in cv_results.items():
        print(f"  {name:25s} | ROC-AUC CV = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── 3. Entraînement final + sauvegarde ────────────────────────────────────
    print("\n=== Entraînement final + sauvegarde des modèles ===")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    trained = train_and_save(models, X_train, y_train, model_dir=models_dir)

    # ── 4. Évaluation ────────────────────────────────────────────────────────
    print("\n=== Évaluation sur le jeu de test ===")
    results_df = evaluate_models(trained, X_test, y_test)
    print(results_df.to_string())

    # ── 5. Visualisations (sauvegardées en PNG) ───────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fig_cm = plot_confusion_matrices(trained, X_test, y_test)
    fig_cm.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=120)

    fig_roc = plot_roc_curves(trained, X_test, y_test)
    fig_roc.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=120)

    fig_fi, _ = plot_feature_importance(trained, FEATURES)
    fig_fi.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=120)

    print(f"\nFigures sauvegardées dans : {results_dir}")
    print("\nPipeline terminé ✓")


if __name__ == '__main__':
    main()
