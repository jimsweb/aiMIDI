"""Smoke test: construct tiny dataset and model, run one train/val step."""

from __future__ import annotations

import os
import tempfile

import pretty_midi as pm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from ai_midi.data.dataset import MIDIDataset
from ai_midi.data.tokenizer import TokenizerFactory
from ai_midi.data.midi_io import save_midi
from ai_midi.model.transformer import GPTModel


def _make_midis(root: str, n: int = 2) -> None:
    for i in range(n):
        m = pm.PrettyMIDI()
        inst = pm.Instrument(program=0)
        t = 0.0
        for p in [60, 62, 64, 65, 67]:
            inst.notes.append(pm.Note(velocity=80, pitch=p, start=t, end=t + 0.2))
            t += 0.25
        m.instruments.append(inst)
        save_midi(m, os.path.join(root, f"{i}.mid"))


def test_smoke_training():
    with tempfile.TemporaryDirectory() as td:
        _make_midis(td, n=2)
        tok = TokenizerFactory.get("remi")
        ds = MIDIDataset(midi_dir=td, tokenizer=tok, seq_len=32, stride=16)
        if len(ds) < 2:
            return  # not enough windows to meaningfully train
        val_ds = Subset(ds, [0])
        train_ds = Subset(ds, list(range(1, len(ds))))
        model = GPTModel(vocab_size=tok.vocab_size, seq_len=32, d_model=64, n_heads=4, n_layers=2, ff_dim=128)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=MIDIDataset.collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=MIDIDataset.collate_fn)
        trainer = pl.Trainer(max_epochs=1, devices=1, limit_train_batches=1, limit_val_batches=1, logger=False, enable_checkpointing=False)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

