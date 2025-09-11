"""Tokenizer encode/decode roundtrip tests."""

from __future__ import annotations

import pretty_midi as pm

from ai_midi.data.tokenizer import RemiTokenizer


def _simple_midi() -> pm.PrettyMIDI:
    m = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    inst.notes.append(pm.Note(velocity=96, pitch=64, start=0.0, end=0.2))
    inst.notes.append(pm.Note(velocity=80, pitch=67, start=0.3, end=0.6))
    m.instruments.append(inst)
    return m


def test_encode_decode_roundtrip():
    tok = RemiTokenizer()
    midi = _simple_midi()
    ids = tok.encode(midi)
    midi2 = tok.decode(ids)
    assert len(midi2.instruments) >= 1
    notes = midi2.instruments[0].notes
    assert len(notes) >= 2
    assert 0 <= notes[0].pitch <= 127
    assert notes[0].end > notes[0].start

