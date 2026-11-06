"""Benchmark inference speed: batch size and FP16.

    python -m src.benchmark --model-dir models/tuned --n 1000

Measures milliseconds PER REVIEW (normalisation + tokenisation + forward pass)
for:
  * FP32, batch size 1     <- naive one-at-a-time serving
  * FP32, batch size 32
  * FP16, batch size 32    <- needs a CUDA GPU (e.g. a free Colab T4)
and checks that FP16 does not hurt macro-F1 on the test split.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from .config import LABELS, RESULTS_DIR
from .data import load_split
from .metrics import compute_metrics
from .predict import SentimentPredictor


def time_it(predictor: SentimentPredictor, texts, batch_size: int, repeats: int = 3) -> float:
    def sync():
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()

    predictor.predict(texts[: min(64, len(texts))], batch_size)  # warm-up
    runs = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        predictor.predict(texts, batch_size)
        sync()
        runs.append((time.perf_counter() - start) * 1000 / len(texts))
    return statistics.median(runs)


def macro_f1(predictor: SentimentPredictor, df, names) -> float:
    label2id = {n: i for i, n in enumerate(names)}
    preds = predictor.predict(df["text"].tolist(), batch_size=64)
    y_pred = [label2id[p["label"]] for p in preds]
    return compute_metrics(df["label_id"].tolist(), y_pred, names)["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--n", type=int, default=1000, help="Number of test reviews to time")
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark.json"))
    args = parser.parse_args()

    base = SentimentPredictor(args.model_dir, fp16=False)
    mode = base.config.get("mode", "three")
    names = LABELS[mode]
    test = load_split(mode, "test")
    texts = test["text"].tolist()[: args.n]
    device = base.device.type
    print(f"Device: {device} | reviews timed per run: {len(texts)}\n")

    rows = [
        ("FP32, batch 1", time_it(base, texts[: min(300, len(texts))], 1)),
        ("FP32, batch 32", time_it(base, texts, 32)),
    ]
    f1 = {"fp32": macro_f1(base, test, names)}

    if device == "cuda":
        fast = SentimentPredictor(args.model_dir, fp16=True)
        rows.append(("FP16, batch 32", time_it(fast, texts, 32)))
        f1["fp16"] = macro_f1(fast, test, names)
    else:
        print("No CUDA GPU found: FP16 benchmark skipped. Run on Colab (T4) to measure it.\n")

    naive = rows[0][1]
    table = [{"setup": n, "ms_per_review": round(ms, 3),
              "reduction_vs_batch1_fp32_pct": round((1 - ms / naive) * 100, 1)} for n, ms in rows]

    print("| Setup | ms / review | Reduction vs FP32 batch 1 |")
    print("|---|---|---|")
    for r in table:
        print(f"| {r['setup']} | {r['ms_per_review']} | {r['reduction_vs_batch1_fp32_pct']}% |")
    print(f"\nTest macro-F1: {f1}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                   "n_reviews": len(texts), "timings": table, "test_macro_f1": f1}, fh, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
