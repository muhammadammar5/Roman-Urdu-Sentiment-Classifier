"""Fine-tune DistilBERT for Roman Urdu sentiment classification (plain PyTorch loop).

    python -m src.train --mode three --run-name tuned
    python -m src.train --mode three --run-name no_norm --no-normalize     # ablation
    python -m src.train --mode three --model distilbert-base-uncased       # other backbone

Features: dynamic padding, mixed-precision (FP16) training on GPU, linear
warm-up/decay schedule, optional class weights, early stopping on validation
macro-F1, and a single final evaluation on the untouched test split.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import DEFAULT_MODEL, LABELS, MAX_LEN, MODELS_DIR, RESULTS_DIR, SEED
from .data import load_split
from .metrics import compute_metrics, plot_confusion_matrix
from .normalize import normalize_many
from .torch_utils import make_loader, predict_logits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(LABELS), default="three")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--class-weights", action="store_true",
                   help="Weight the loss inversely to class frequency")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed-precision training (GPU only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    names = LABELS[args.mode]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and device.type == "cuda"
    run_name = args.run_name or f"{args.model.split('/')[-1]}_{args.mode}_{'norm' if args.normalize else 'raw'}"
    out_dir = MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run '{run_name}' on {device} | amp={use_amp} | model={args.model}")

    prep = normalize_many if args.normalize else (lambda xs: [str(x).strip() for x in xs])
    train_df, val_df, test_df = (load_split(args.mode, s) for s in ("train", "val", "test"))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_loader = make_loader(tokenizer, prep(train_df["text"]), train_df["label_id"],
                               args.batch_size, args.max_len, shuffle=True)
    val_loader = make_loader(tokenizer, prep(val_df["text"]), val_df["label_id"],
                             128, args.max_len, shuffle=False)
    test_loader = make_loader(tokenizer, prep(test_df["text"]), test_df["label_id"],
                              128, args.max_len, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
    ).to(device)

    weights = None
    if args.class_weights:
        counts = np.bincount(train_df["label_id"], minlength=len(names))
        weights = torch.tensor(len(train_df) / (len(names) * counts), dtype=torch.float, device=device)
        print(f"Class weights: {dict(zip(names, weights.tolist()))}")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history, best_f1, bad_epochs = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = (t.to(device) for t in (input_ids, attention_mask, labels))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        logits, labels = predict_logits(model, val_loader, device, amp=use_amp)
        val_loss = nn.functional.cross_entropy(logits, labels.long()).item()  # unweighted, for reporting
        val_metrics = compute_metrics(labels.tolist(), logits.argmax(-1).tolist(), names)
        record = {"epoch": epoch, "train_loss": round(running / seen, 4),
                  "val_loss": round(val_loss, 4), "val_macro_f1": val_metrics["macro_f1"],
                  "val_accuracy": val_metrics["accuracy"]}
        history.append(record)
        print(record)

        if val_metrics["macro_f1"] > best_f1:
            best_f1, bad_epochs = val_metrics["macro_f1"], 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    train_seconds = round(time.perf_counter() - start, 1)

    # Final, single look at the test set using the best checkpoint
    best_model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    logits, labels = predict_logits(best_model, test_loader, device, amp=False)
    test_metrics = compute_metrics(labels.tolist(), logits.argmax(-1).tolist(), names)

    (out_dir / "rus_config.json").write_text(json.dumps({
        "normalize": args.normalize, "max_len": args.max_len, "mode": args.mode,
        "labels": names, "base_model": args.model,
    }, indent=2), encoding="utf-8")

    RESULTS_DIR.mkdir(exist_ok=True)
    result = {
        "run_name": run_name, "kind": "transformer", "model": args.model,
        "mode": args.mode, "normalize": args.normalize, "seed": args.seed,
        "hyperparameters": {k: getattr(args, k) for k in
                            ("epochs", "batch_size", "lr", "weight_decay", "max_len",
                             "class_weights", "fp16")},
        "val_macro_f1": best_f1,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "epochs_trained": len(history), "train_seconds": train_seconds,
        "history": history, "test": test_metrics,
    }
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_confusion_matrix(test_metrics["confusion_matrix"], names,
                          RESULTS_DIR / f"{run_name}_confusion.png", f"{run_name} (test)")
    print(f"\n[{run_name}] TEST accuracy={test_metrics['accuracy']:.4f} "
          f"macro-F1={test_metrics['macro_f1']:.4f}  | model saved to {out_dir}")


if __name__ == "__main__":
    main()
