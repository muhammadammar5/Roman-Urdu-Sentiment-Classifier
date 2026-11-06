"""Evaluation metrics shared by the baseline, trainer and evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                    label_names: List[str]) -> Dict[str, object]:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(label_names)
        },
        "labels": label_names,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def plot_confusion_matrix(cm: List[List[int]], labels: List[str], path: Path,
                          title: str = "Confusion matrix") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
