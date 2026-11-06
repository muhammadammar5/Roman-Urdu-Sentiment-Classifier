"""Print a Markdown table of every result in results/*.json.

    python -m src.compare
"""
from __future__ import annotations

import json

from .config import RESULTS_DIR


def main() -> None:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "test" in data and "run_name" in data:
            runs.append(data)
    if not runs:
        raise SystemExit("No results found. Run src.baseline and src.train first.")

    runs.sort(key=lambda r: r["test"]["macro_f1"])
    print("| Run | Model | Classes | Normalised | Accuracy | Macro-F1 |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['run_name']} | {r['model']} | {r['mode']} | "
              f"{'yes' if r['normalize'] else 'no'} | {r['test']['accuracy']:.4f} | "
              f"{r['test']['macro_f1']:.4f} |")

    # Normalisation effect: same model family and mode, normalised vs raw
    groups = {}
    for r in runs:
        groups.setdefault((r["kind"], r["model"].split(" (")[0], r["mode"]), {})[r["normalize"]] = r
    print()
    for (kind, model, mode), pair in groups.items():
        if True in pair and False in pair:
            delta = (pair[True]["test"]["macro_f1"] - pair[False]["test"]["macro_f1"]) * 100
            print(f"Normalisation effect on {model} ({mode}): {delta:+.1f} macro-F1 points "
                  f"(single run each; differences under ~1 point are within seed noise)")


if __name__ == "__main__":
    main()
