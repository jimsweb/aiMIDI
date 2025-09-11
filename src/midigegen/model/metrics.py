"""Basic metrics helpers."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = labels.ne(ignore_index)
        correct = (preds.eq(labels) & mask).sum()
        total = mask.sum().clamp_min(1)
        return (correct.float() / total.float())

