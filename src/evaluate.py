"""Evaluate a saved model on a split through the same code path used in deployment.

    python -m src.evaluate --model-dir models/tuned --split test
"""
from __future__ import annotations

import argparse
import json

from .config import LABELS, RESULTS_DIR
from .data import load_split
from .metrics import compute_metrics, plot_confusion_matrix
from .predict import SentimentPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir, fp16=args.fp16)
    mode = predictor.config.get("mode", "three")
    names = LABELS[mode]
    df = load_split(mode, args.split)

    preds = predictor.predict(df["text"].tolist(), batch_size=64)
    label2id = {n: i for i, n in enumerate(names)}
    y_pred = [label2id[p["label"]] for p in preds]
    metrics = compute_metrics(df["label_id"].tolist(), y_pred, names)

    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class")}, indent=2))
    tag = args.model_dir.rstrip("/").split("/")[-1]
    plot_confusion_matrix(metrics["confusion_matrix"], names,
                          RESULTS_DIR / f"{tag}_{args.split}_confusion.png",
                          f"{tag} ({args.split})")


if __name__ == "__main__":
    main()
