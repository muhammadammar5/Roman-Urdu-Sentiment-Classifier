"""Upload a trained model to the Hugging Face Hub (needed by the Space demo).

    huggingface-cli login            # or: hf auth login   (needs a *write* token)
    python -m src.push_to_hub --model-dir models/tuned --repo-id <hf-username>/roman-urdu-sentiment-distilbert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
language:
- ur
license: mit
library_name: transformers
pipeline_tag: text-classification
tags:
- sentiment-analysis
- roman-urdu
- distilbert
---

# Roman Urdu Sentiment Classifier

DistilBERT fine-tuned on the Roman Urdu Data Set (Sharf, 2017, UCI ML Repository, CC BY 4.0)
to classify Roman Urdu / English code-mixed text as {labels}.

{metrics}
Preprocessing matters: inputs were passed through a Roman Urdu normaliser
(see `rus_config.json` and the project repository) before tokenisation.

Source code, training and evaluation: https://github.com/muhammadammar5/Roman-Urdu-Sentiment-Classifier
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "rus_config.json").read_text(encoding="utf-8"))

    metrics = ""
    results = Path("results") / f"{model_dir.name}.json"
    if results.exists():
        test = json.loads(results.read_text(encoding="utf-8"))["test"]
        metrics = (f"Test set: accuracy {test['accuracy']:.3f}, macro-F1 {test['macro_f1']:.3f}.\n")

    (model_dir / "README.md").write_text(
        CARD.format(labels=", ".join(cfg["labels"]), metrics=metrics), encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo_id)
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
