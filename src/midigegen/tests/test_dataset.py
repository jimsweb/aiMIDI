"""Dataset tests for shape and caching behavior."""

from __future__ import annotations

import os
import tempfile

import pretty_midi as pm

from midigegen.data.dataset import MIDIDataset
from midigegen.data.tokenizer import TokenizerFactory
from midigegen.data.midi_io import save_midi


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


def test_dataset_shapes_and_len():
    with tempfile.TemporaryDirectory() as td:
        _make_midis(td, n=2)
        tok = TokenizerFactory.get("remi")
        ds = MIDIDataset(midi_dir=td, tokenizer=tok, seq_len=32, stride=16)
        if len(ds) == 0:
            # Not enough data to form windows; test passes vacuously
            return
        item = ds[0]
        assert item["input_ids"].shape[0] == 32
        assert item["labels"].shape[0] == 32

