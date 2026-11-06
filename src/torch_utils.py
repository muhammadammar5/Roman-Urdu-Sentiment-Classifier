"""Small PyTorch helpers: tokenised dataset, dynamic-padding collate, batched inference."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


class IdsDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], labels: Optional[Sequence[int]] = None):
        self.input_ids = input_ids
        self.labels = list(labels) if labels is not None else [-1] * len(input_ids)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.labels[idx]


def make_collate(pad_id: int):
    """Pad each batch only to its own longest sequence (dynamic padding)."""

    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return input_ids, attention_mask, labels

    return collate


def make_loader(tokenizer, texts: Sequence[str], labels: Sequence[int],
                batch_size: int, max_len: int, shuffle: bool) -> DataLoader:
    ids = tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]
    return DataLoader(
        IdsDataset(ids, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )


@torch.inference_mode()
def predict_logits(model, loader: DataLoader, device: torch.device, amp: bool = False):
    """Run the model over a loader and return (logits, labels) as CPU tensors."""
    model.eval()
    all_logits, all_labels = [], []
    use_amp = amp and device.type == "cuda"
    for input_ids, attention_mask, labels in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)
