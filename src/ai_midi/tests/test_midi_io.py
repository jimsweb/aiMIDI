"""Tests for MIDI IO utilities.

Creates a simple PrettyMIDI with one note, saves, loads, and checks fields.
"""

from __future__ import annotations

import os
import tempfile

import pretty_midi as pm

from ai_midi.data.midi_io import load_midi, save_midi


def _one_note() -> pm.PrettyMIDI:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.notes.append(pm.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    m.instruments.append(inst)
    return m


def test_load_save_load_roundtrip():
    m = _one_note()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.mid")
        save_midi(m, p)
        m2 = load_midi(p)
        assert len(m2.instruments) == 1
        assert len(m2.instruments[0].notes) == 1
        n = m2.instruments[0].notes[0]
        assert n.pitch == 60
        assert n.velocity == 100

