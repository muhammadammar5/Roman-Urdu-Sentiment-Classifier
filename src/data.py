"""Download, clean and split the Roman Urdu sentiment dataset.

    python -m src.data --download --mode three     # 3-class: Negative/Neutral/Positive
    python -m src.data --mode binary               # drop Neutral, Negative vs Positive

Cleaning (all counts are written to data/processed/<mode>/stats.json):
  * fix the known "Neative" label typo; drop rows with missing text or bad labels
  * drop texts that appear with CONFLICTING labels (ambiguous)
  * de-duplicate on the NORMALISED text, so near-identical copies cannot leak
    across train/validation/test
  * stratified 80/10/10 split with a fixed seed
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LABELS, PROCESSED_DIR, RAW_PATH, SEED, UCI_URL
from .normalize import normalize_text

LABEL_FIXES = {"neative": "Negative", "negative": "Negative",
               "positive": "Positive", "neutral": "Neutral"}


def download_raw(dest: Path = RAW_PATH, url: str = UCI_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the UCI CSV (no header, text,label,<empty>) into columns text,label."""
    df = pd.read_csv(
        path,
        header=None,
        names=["text", "label", "extra"],
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip",
    )
    return df[["text", "label"]]


def clean(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats: Dict[str, int] = {"raw_rows": len(df)}

    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_FIXES)
    df = df.dropna(subset=["label"])
    stats["after_label_fix"] = len(df)

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["key"] = df["text"].map(normalize_text)
    df = df[df["key"].str.len() > 0]
    stats["after_dropping_empty_text"] = len(df)

    # texts whose duplicates disagree on the label are ambiguous -> drop them all
    label_count = df.groupby("key")["label"].nunique()
    conflicting = label_count[label_count > 1].index
    df = df[~df["key"].isin(conflicting)]
    stats["dropped_conflicting_label_texts"] = int(len(conflicting))

    df = df.drop_duplicates(subset="key", keep="first")
    stats["after_deduplication"] = len(df)

    if mode == "binary":
        df = df[df["label"] != "Neutral"]
    stats["final_rows"] = len(df)

    names = LABELS[mode]
    df["label_id"] = df["label"].map({n: i for i, n in enumerate(names)})
    return df[["text", "label", "label_id"]].reset_index(drop=True), stats


def split(df: pd.DataFrame, seed: int = SEED, val_size: float = 0.1,
          test_size: float = 0.1):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label_id"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR,
            mode: str = "three", seed: int = SEED) -> Dict[str, object]:
    df, stats = clean(load_raw(raw_path), mode)
    train, val, test = split(df, seed)

    target = Path(out_dir) / mode
    target.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(target / f"{name}.csv", index=False)

    stats.update(
        mode=mode,
        seed=seed,
        splits={"train": len(train), "val": len(val), "test": len(test)},
        class_counts=df["label"].value_counts().to_dict(),
    )
    (target / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_split(mode: str, split_name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = Path(processed_dir) / mode / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data --download --mode {mode}"
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Roman Urdu dataset")
    parser.add_argument("--download", action="store_true", help="Download the CSV from UCI first")
    parser.add_argument("--raw-path", default=str(RAW_PATH))
    parser.add_argument("--mode", choices=list(LABELS), default="three")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    raw = Path(args.raw_path)
    if args.download or not raw.exists():
        download_raw(raw)
    stats = prepare(raw, PROCESSED_DIR, args.mode, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
