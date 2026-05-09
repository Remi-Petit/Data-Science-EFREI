"""
Lance les train_pipeline.py des sujets sélectionnés.

Usage :
    python train_all.py          # relance tous les sujets
    python train_all.py 1 3      # relance uniquement Sujet_1 et Sujet_3
"""
import os
import sys
import subprocess

SUJETS = {
    1: os.path.join(os.path.dirname(__file__), 'Sujet_1'),
    2: os.path.join(os.path.dirname(__file__), 'Sujet_2'),
    3: os.path.join(os.path.dirname(__file__), 'Sujet_3'),
}


def run(sujet_id: int, path: str) -> bool:
    print(f"\n{'='*55}")
    print(f"  Sujet {sujet_id} – {path}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, 'train_pipeline.py'],
        cwd=path,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
    )
    if result.returncode != 0:
        print(f"\n[ERREUR] Sujet {sujet_id} a échoué (code {result.returncode})")
        return False
    return True


def main():
    args = sys.argv[1:]
    if args:
        try:
            selected = [int(a) for a in args]
        except ValueError:
            print("Usage : python train_all.py [1] [2] [3]")
            sys.exit(1)
        unknown = [s for s in selected if s not in SUJETS]
        if unknown:
            print(f"Sujets inconnus : {unknown}. Valeurs valides : {list(SUJETS.keys())}")
            sys.exit(1)
    else:
        selected = list(SUJETS.keys())

    failures = []
    for sid in selected:
        ok = run(sid, SUJETS[sid])
        if not ok:
            failures.append(sid)

    print(f"\n{'='*55}")
    if failures:
        print(f"  Terminé avec erreurs sur les sujets : {failures}")
    else:
        print(f"  Tous les pipelines ont réussi ({selected}).")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
