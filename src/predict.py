"""Inference wrapper used by the evaluator, the benchmark and the CLI.

    python -m src.predict --model-dir models/tuned "yeh phone bohat acha hai"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_LEN
from .normalize import normalize_text


def load_run_config(model_id: str) -> Dict[str, object]:
    """Read rus_config.json from a local folder or a Hugging Face Hub repo."""
    local = Path(model_id) / "rus_config.json"
    if Path(model_id).is_dir():
        return json.loads(local.read_text(encoding="utf-8")) if local.exists() else {}
    try:
        from huggingface_hub import hf_hub_download

        return json.loads(Path(hf_hub_download(model_id, "rus_config.json")).read_text(encoding="utf-8"))
    except Exception:
        return {}


class SentimentPredictor:
    def __init__(self, model_id: str, device: Optional[str] = None, fp16: bool = False,
                 normalize: Optional[bool] = None, max_len: Optional[int] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = load_run_config(model_id)
        self.config = cfg
        self.normalize = cfg.get("normalize", True) if normalize is None else normalize
        self.max_len = max_len or cfg.get("max_len", MAX_LEN)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        if self.fp16:
            self.model.half()
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, object]]:
        """Classify texts. Inputs are length-sorted before batching so each batch
        needs minimal padding; results are returned in the original order."""
        prepared = [normalize_text(t) if self.normalize else str(t).strip() for t in texts]
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        results: List[Optional[Dict[str, object]]] = [None] * len(prepared)

        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = self.tokenizer(
                [prepared[i] for i in idx], truncation=True, max_length=self.max_len,
                padding=True, return_tensors="pt", return_token_type_ids=False,
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
            for row, i in zip(probs, idx):
                scores = {self.id2label[j]: float(row[j]) for j in range(row.numel())}
                label = max(scores, key=scores.get)
                results[i] = {"label": label, "confidence": scores[label], "scores": scores}
        return results  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Local folder or Hub repo id")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir, fp16=args.fp16)
    for text, out in zip(args.texts, predictor.predict(args.texts)):
        print(f"{out['label']:<9} {out['confidence']:.2%}  |  {text}")


if __name__ == "__main__":
    main()
