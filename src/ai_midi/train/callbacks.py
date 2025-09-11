"""Training callbacks for checkpointing and sample generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from ..infer.generator import sample_and_save


class SaveBestModelCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, dirpath: str, monitor: str = "val/loss", mode: str = "min") -> None:
        super().__init__(dirpath=dirpath, monitor=monitor, mode=mode, save_top_k=1, filename="best")


class SampleGenerationCallback(pl.Callback):
    def __init__(self, samples_dir: str, every_n_epochs: int = 1, max_tokens: int = 128) -> None:
        super().__init__()
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = every_n_epochs
        self.max_tokens = max_tokens

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        try:
            out_path = self.samples_dir / f"sample_epoch{epoch+1}.mid"
            sample_and_save(pl_module, out_path=str(out_path), max_tokens=self.max_tokens)
        except Exception as e:  # pragma: no cover - sampling failures should not crash training
            pl_module.print(f"SampleGenerationCallback failed: {e}")

