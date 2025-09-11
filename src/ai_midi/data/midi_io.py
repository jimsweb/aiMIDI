"""MIDI IO utilities using pretty_midi with deterministic quantization.

Functions
---------
- load_midi(path): Load a MIDI as pretty_midi.PrettyMIDI.
- save_midi(pm, path): Save PrettyMIDI to path.
- extract_notes(pm, instrument): Extract (start, end, pitch, velocity, program).
- normalize_tempo(pm, target_bpm): Replace tempo changes with a single target tempo.
- time/grid helpers: seconds<->steps via fixed base_time_ms; steps<->ticks via BPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi as pm


def load_midi(path: str) -> pm.PrettyMIDI:
    """Load a MIDI file.

    Parameters
    ----------
    path: str
        Path to MIDI file.
    """

    return pm.PrettyMIDI(path)


def save_midi(m: pm.PrettyMIDI, path: str) -> None:
    """Save a PrettyMIDI object to disk."""

    m.write(path)


def normalize_tempo(m: pm.PrettyMIDI, target_bpm: float = 120.0) -> None:
    """Normalize a MIDI's tempo map to a single BPM.

    This modifies the PrettyMIDI object in-place by setting a single tempo event
    at time 0.
    """

    m.remove_tempo_changes()
    m._tick_scales = None  # ensure recomputation
    tempo = np.array([target_bpm], dtype=float)
    time = np.array([0.0], dtype=float)
    m._update_tick_to_time(tempo, time)


def extract_notes(m: pm.PrettyMIDI, instrument: Optional[int] = None) -> List[Tuple[float, float, int, int, int]]:
    """Extract notes from PrettyMIDI as tuples.

    Returns list of (start_time, end_time, pitch, velocity, program).
    If `instrument` is provided, filters to that program number.
    """

    notes: List[Tuple[float, float, int, int, int]] = []
    for inst in m.instruments:
        if instrument is not None and inst.program != instrument:
            continue
        for n in inst.notes:
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity), int(inst.program)))
    notes.sort(key=lambda x: (x[0], x[2]))
    return notes


@dataclass(frozen=True)
class QuantizationCfg:
    base_time_ms: int = 20
    steps_per_bar: int = 64
    max_duration_steps: int = 128

    @property
    def step_seconds(self) -> float:
        return self.base_time_ms / 1000.0


def seconds_to_steps(t: float, q: QuantizationCfg) -> int:
    return int(round(t / q.step_seconds))


def steps_to_seconds(steps: int, q: QuantizationCfg) -> float:
    return steps * q.step_seconds


def quantize_time(t: float, q: QuantizationCfg) -> float:
    return steps_to_seconds(seconds_to_steps(t, q), q)

