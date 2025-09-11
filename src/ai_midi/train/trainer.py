"""Training orchestration using PyTorch Lightning and Hydra config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from omegaconf import DictConfig

from ..data.dataset import MIDIDataset
from ..data.tokenizer import TokenizerFactory, RemiTokenizer
from ..model.transformer import GPTModel
from ..train.callbacks import SaveBestModelCallback, SampleGenerationCallback
from ..utils.io import ensure_dir
from ..utils.seed import seed_everything


def run_train(cfg: DictConfig) -> None:
    seed_everything(int(cfg.get("seed", 42)))

    # Tokenizer
    tokenizer = TokenizerFactory.get("remi")
    vocab_size = tokenizer.vocab_size

    # Datasets
    data_cfg = cfg.data
    train_ds = MIDIDataset(
        midi_dir=data_cfg.data_dir,
        tokenizer=tokenizer,
        cache_dir=data_cfg.cache_dir,
        seq_len=data_cfg.seq_len,
        stride=data_cfg.stride,
        min_len=data_cfg.min_len,
    )
    # For simplicity, reuse a small subset as val
    n_val = max(1, len(train_ds) // 10)
    val_indices = list(range(0, n_val))
    train_indices = list(range(n_val, len(train_ds)))
    from torch.utils.data import Subset

    val_ds = Subset(train_ds, val_indices)
    train_ds2 = Subset(train_ds, train_indices)

    # Model
    model_cfg = cfg.model
    model = GPTModel(
        vocab_size=vocab_size,
        seq_len=model_cfg.seq_len,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        ff_dim=model_cfg.ff_dim,
        dropout=model_cfg.dropout,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        warmup_steps=cfg.train.warmup_steps,
        pad_id=0,
    )

    # Loaders
    dl_kwargs = dict(
        batch_size=cfg.train.batch_size,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        collate_fn=MIDIDataset.collate_fn,
    )
    train_loader = DataLoader(train_ds2, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # Logging and callbacks
    exp_name = cfg.get("experiment_name", "run")
    ckpt_dir = ensure_dir(cfg.train.checkpoint_dir)
    samples_dir = ensure_dir(cfg.train.samples_dir)
    callbacks = [
        SaveBestModelCallback(dirpath=str(ckpt_dir)),
        SampleGenerationCallback(samples_dir=str(samples_dir), every_n_epochs=1, max_tokens=128),
    ]
    logger = CSVLogger(save_dir="logs", name=exp_name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        devices=int(cfg.train.devices),
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

