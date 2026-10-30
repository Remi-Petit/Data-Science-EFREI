"""
Entraîne et sauvegarde (.joblib) les pipelines de chaque sujet, sans cross-validation
ni évaluation (les stats versionnées dans models_stats/ ne sont pas recalculées).

Usage :
    python IA/train_models.py          # tous les sujets
    python IA/train_models.py 1 3      # sujets choisis
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUJETS = (1, 2, 3)


def train(sujet: int) -> None:
    base = os.path.join(HERE, f"Sujet_{sujet}")
    sys.path.insert(0, base)
    os.chdir(base)
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)

    from src.preprocessing import load_data, engineer_features, get_train_test_split

    df = engineer_features(load_data())

    if sujet == 1:
        from src.preprocessing import get_type_train_test_split, get_rul_train_test_split
        from src.train import build_pipelines, train_and_save, train_and_save_type, train_and_save_rul
        X, _, y, _ = get_train_test_split(df)
        train_and_save(build_pipelines(), X, y, model_dir=models_dir)
        X, _, y, _ = get_type_train_test_split(df)
        train_and_save_type(X, y, model_dir=models_dir)
        X, _, y, _ = get_rul_train_test_split(df)
        train_and_save_rul(X, y, model_dir=models_dir)
    elif sujet == 2:
        from src.train import build_pipelines, train_and_save
        X, _, y, _ = get_train_test_split(df)
        train_and_save(build_pipelines(), X, y, model_dir=models_dir)
    else:
        from src.train import build_regression_pipelines, train_and_save, MODEL_FILENAMES_REG
        X, _, y_reg, _, _, _ = get_train_test_split(df)
        train_and_save(build_regression_pipelines(), X, y_reg, MODEL_FILENAMES_REG, models_dir)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--subject":
        train(int(args[1]))
        return
    selected = [int(a) for a in args] or list(SUJETS)
    unknown = [s for s in selected if s not in SUJETS]
    if unknown:
        sys.exit(f"Sujets inconnus : {unknown}. Valeurs valides : {list(SUJETS)}")
    for sid in selected:
        print(f"== Sujet {sid}", flush=True)
        # Un process par sujet : chaque sujet a son propre package `src`.
        subprocess.run([sys.executable, __file__, "--subject", str(sid)], check=True)


if __name__ == "__main__":
    main()
