"""TF-IDF + linear SVM baseline.

    python -m src.baseline --mode three
    python -m src.baseline --mode three --no-normalize

Word (1-2 gram) and character (2-5 gram) TF-IDF features are combined, which
copes well with Roman Urdu's spelling noise. The regularisation strength C is
chosen on the validation split; the test split is touched once, at the end.
"""
from __future__ import annotations

import argparse
import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import LABELS, RESULTS_DIR, SEED
from .data import load_split
from .metrics import compute_metrics
from .normalize import normalize_many


def build_pipeline(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True)),
    ])
    return Pipeline([
        ("tfidf", features),
        ("svm", LinearSVC(C=C, class_weight="balanced", random_state=SEED)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(LABELS), default="three")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    names = LABELS[args.mode]
    prep = normalize_many if args.normalize else (lambda xs: [str(x).strip().lower() for x in xs])
    train, val, test = (load_split(args.mode, s) for s in ("train", "val", "test"))
    X_train, X_val, X_test = (prep(d["text"]) for d in (train, val, test))

    start = time.perf_counter()
    best = None
    for C in (0.1, 0.3, 1.0, 3.0):
        model = build_pipeline(C).fit(X_train, train["label_id"])
        val_f1 = compute_metrics(val["label_id"], model.predict(X_val), names)["macro_f1"]
        print(f"C={C}: val macro-F1 = {val_f1:.4f}")
        if best is None or val_f1 > best[0]:
            best = (val_f1, C, model)
    val_f1, best_C, model = best

    test_metrics = compute_metrics(test["label_id"], model.predict(X_test), names)
    run_name = args.run_name or f"baseline_{args.mode}_{'norm' if args.normalize else 'raw'}"
    result = {
        "run_name": run_name,
        "kind": "baseline",
        "model": f"TF-IDF + LinearSVC (C={best_C})",
        "mode": args.mode,
        "normalize": args.normalize,
        "val_macro_f1": val_f1,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_seconds": round(time.perf_counter() - start, 1),
        "test": test_metrics,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[{run_name}] TEST accuracy={test_metrics['accuracy']:.4f} "
          f"macro-F1={test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
