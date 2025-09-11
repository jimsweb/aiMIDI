"""PyTorch Dataset for MIDI token sequences with on-disk caching.

The dataset accepts a directory of MIDI files (or a list of files) and a
Tokenizer. It encodes each MIDI to a token sequence, caches the arrays as .npy,
and serves sliding windows of length `seq_len` with stride `stride`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .midi_io import load_midi
from .tokenizer import Tokenizer
from ..utils.io import ensure_dir


class MIDIDataset(Dataset):
    """Dataset producing (input_ids, labels) pairs for LM training."""

    def __init__(
        self,
        midi_dir: str | None = None,
        files: Optional[Sequence[str]] = None,
        tokenizer: Optional[Tokenizer] = None,
        cache_dir: Optional[str] = None,
        seq_len: int = 512,
        stride: int = 256,
        min_len: int = 32,
        reprocess: bool = False,
    ) -> None:
        assert (midi_dir is not None) ^ (files is not None), "Provide either midi_dir or files"
        self.root = Path(midi_dir) if midi_dir is not None else None
        self.files = list(files) if files is not None else self._scan(self.root)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.min_len = min_len
        self.cache_dir = Path(cache_dir) if cache_dir else (self.root / "_cache" if self.root else Path("_cache"))
        ensure_dir(self.cache_dir)
        # Process and cache
        self._all_tokens: List[np.ndarray] = []
        self._index: List[Tuple[int, int]] = []  # (sequence_idx, start)
        self._prepare_cache(reprocess)

    def _scan(self, root: Path) -> List[str]:
        return [str(p) for p in sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))]

    def _hash(self, path: str) -> str:
        return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, src: str) -> Path:
        return self.cache_dir / f"{self._hash(src)}.npy"

    def _prepare_cache(self, reprocess: bool) -> None:
        assert self.tokenizer is not None, "Tokenizer is required"
        for f in self.files:
            cpath = self._cache_path(f)
            if cpath.exists() and not reprocess:
                arr = np.load(cpath)
            else:
                m = load_midi(f)
                tok = np.array(self.tokenizer.encode(m), dtype=np.int64)
                np.save(cpath, tok)
                arr = tok
            if arr.shape[0] >= self.min_len:
                self._all_tokens.append(arr)
        # Build index for sliding windows
        for i, arr in enumerate(self._all_tokens):
            L = len(arr)
            start = 0
            while start + self.seq_len < L:
                self._index.append((i, start))
                start += self.stride

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx, start = self._index[idx]
        arr = self._all_tokens[seq_idx]
        x = arr[start : start + self.seq_len]
        y = arr[start + 1 : start + self.seq_len + 1]
        # pad if needed
        pad_id = 0  # PAD is id 0 in our vocab build
        if len(y) < self.seq_len:
            pad_len = self.seq_len - len(y)
            x = np.pad(x, (0, pad_len), constant_values=pad_id)
            y = np.pad(y, (0, pad_len), constant_values=pad_id)
        return {
            "input_ids": torch.from_numpy(x.astype(np.int64)),
            "labels": torch.from_numpy(y.astype(np.int64)),
        }

    @staticmethod
    def collate_fn(batch: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"input_ids": input_ids, "labels": labels}

