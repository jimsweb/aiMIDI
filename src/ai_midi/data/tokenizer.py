"""Tokenizer implementations for MIDI events (REMI-like).

We implement a simple combined-note scheme:
- TimeShift<N>: move time forward by N quantized steps.
- Velocity<V>: set active velocity bin for next notes.
- Program<P>: set active program (instrument) for next notes.
- Note_<pitch>_Dur_<d>: a single event that starts a note at current time with
  pitch in [0,127] and duration of d steps (clipped to max_duration_steps).
- Special tokens: SOS, EOS, PAD.

The tokenizer maintains a vocabulary of these events and provides encode/decode
to and from pretty_midi.PrettyMIDI objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pretty_midi as pm

from .midi_io import QuantizationCfg, extract_notes, seconds_to_steps, steps_to_seconds
from ..utils.io import load_json, save_json

SPECIAL = ["PAD", "SOS", "EOS"]


class Tokenizer:
    """Abstract base class for MIDI tokenizers."""

    def encode(self, midi: pm.PrettyMIDI) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: List[int]) -> pm.PrettyMIDI:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def save_vocab(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load_vocab(cls, path: str | Path) -> "Tokenizer":  # pragma: no cover - factory path
        raise NotImplementedError


@dataclass
class RemiTokenizerCfg:
    quantization_steps_per_bar: int = 64
    base_time_ms: int = 20
    velocity_bins: int = 32
    max_duration_steps: int = 128


class RemiTokenizer(Tokenizer):
    """Simple REMI-like tokenizer with combined note events.

    Configurable via RemiTokenizerCfg. Vocabulary is generated on init.
    """

    def __init__(self, cfg: Optional[RemiTokenizerCfg] = None) -> None:
        self.cfg = cfg or RemiTokenizerCfg()
        self.q = QuantizationCfg(
            base_time_ms=self.cfg.base_time_ms,
            steps_per_bar=self.cfg.quantization_steps_per_bar,
            max_duration_steps=self.cfg.max_duration_steps,
        )
        # Build vocabulary
        self.event_to_id: Dict[str, int] = {}
        self.id_to_event: Dict[int, str] = {}
        self._build_vocab()
        # Defaults
        self._current_velocity_bin = self.cfg.velocity_bins // 2
        self._current_program = 0

    # Vocabulary
    def _add(self, token: str) -> None:
        if token not in self.event_to_id:
            idx = len(self.event_to_id)
            self.event_to_id[token] = idx
            self.id_to_event[idx] = token

    def _build_vocab(self) -> None:
        for s in SPECIAL:
            self._add(s)
        # Velocity bins
        for v in range(self.cfg.velocity_bins):
            self._add(f"Velocity_{v}")
        # Program changes (0-127)
        for p in range(128):
            self._add(f"Program_{p}")
        # TimeShift up to steps_per_bar (repeat as needed during encode)
        for n in range(1, self.cfg.quantization_steps_per_bar + 1):
            self._add(f"TimeShift_{n}")
        # Notes and durations up to max_duration_steps
        for pitch in range(128):
            for d in range(1, self.cfg.max_duration_steps + 1):
                self._add(f"Note_{pitch}_Dur_{d}")

    # Public API
    @property
    def vocab_size(self) -> int:
        return len(self.event_to_id)

    def _vel_to_bin(self, vel: int) -> int:
        # Map 1..127 to 0..(bins-1)
        return min(self.cfg.velocity_bins - 1, max(0, int((vel - 1) / (127 / self.cfg.velocity_bins))))

    def _bin_to_vel(self, vbin: int) -> int:
        # Map bin back to representative velocity
        step = 127 / self.cfg.velocity_bins
        v = int(vbin * step + step / 2)
        return min(127, max(1, v))

    def _append_timeshift(self, events: List[int], delta_steps: int) -> None:
        # Use chunks of at most steps_per_bar
        base = self.cfg.quantization_steps_per_bar
        while delta_steps > 0:
            n = min(base, delta_steps)
            events.append(self.event_to_id[f"TimeShift_{n}"])
            delta_steps -= n

    def encode(self, midi: pm.PrettyMIDI) -> List[int]:
        notes = extract_notes(midi)
        events: List[int] = [self.event_to_id["SOS"]]
        t_cur_steps = 0
        cur_prog = None
        cur_vel_bin = None
        for (start, end, pitch, velocity, program) in notes:
            start_s = seconds_to_steps(start, self.q)
            end_s = seconds_to_steps(end, self.q)
            dur = max(1, min(self.cfg.max_duration_steps, end_s - start_s))
            # timeshift
            self._append_timeshift(events, max(0, start_s - t_cur_steps))
            t_cur_steps = start_s
            # program change if needed
            p = program
            if cur_prog != p:
                events.append(self.event_to_id[f"Program_{p}"])
                cur_prog = p
            # velocity
            vbin = self._vel_to_bin(velocity)
            if cur_vel_bin != vbin:
                events.append(self.event_to_id[f"Velocity_{vbin}"])
                cur_vel_bin = vbin
            # note
            events.append(self.event_to_id[f"Note_{pitch}_Dur_{dur}"])
        events.append(self.event_to_id["EOS"])
        return events

    def decode(self, ids: List[int]) -> pm.PrettyMIDI:
        m = pm.PrettyMIDI()
        insts: Dict[int, pm.Instrument] = {}
        t_steps = 0
        cur_vel_bin = self.cfg.velocity_bins // 2
        cur_prog = 0
        for idx in ids:
            token = self.id_to_event.get(int(idx), "")
            if token in ("PAD", "SOS"):
                continue
            if token == "EOS":
                break
            if token.startswith("TimeShift_"):
                n = int(token.split("_")[-1])
                t_steps += n
            elif token.startswith("Velocity_"):
                cur_vel_bin = int(token.split("_")[-1])
            elif token.startswith("Program_"):
                cur_prog = int(token.split("_")[-1])
                if cur_prog not in insts:
                    insts[cur_prog] = pm.Instrument(program=cur_prog)
            elif token.startswith("Note_"):
                _, rest = token.split("Note_")
                pitch_str, dur_str = rest.split("_Dur_")
                pitch = int(pitch_str)
                dur = int(dur_str)
                start = steps_to_seconds(t_steps, self.q)
                end = steps_to_seconds(t_steps + dur, self.q)
                if cur_prog not in insts:
                    insts[cur_prog] = pm.Instrument(program=cur_prog)
                velocity = self._bin_to_vel(cur_vel_bin)
                insts[cur_prog].notes.append(pm.Note(velocity=velocity, pitch=pitch, start=start, end=end))
            else:
                # unknown token, skip
                continue
        for inst in insts.values():
            m.instruments.append(inst)
        return m

    def save_vocab(self, path: str | Path) -> None:
        save_json({"event_to_id": self.event_to_id, "id_to_event": self.id_to_event}, path)

    @classmethod
    def load_vocab(cls, path: str | Path) -> "RemiTokenizer":
        data = load_json(path)
        tok = cls()
        tok.event_to_id = {str(k): int(v) for k, v in data["event_to_id"].items()}
        tok.id_to_event = {int(k): str(v) for k, v in data["id_to_event"].items()}
        return tok


class TokenizerFactory:
    """Simple factory for tokenizers."""

    @staticmethod
    def get(name: str, **kwargs) -> Tokenizer:
        name = name.lower()
        if name in {"remi", "default"}:
            cfg = kwargs.get("cfg")
            return RemiTokenizer(cfg=cfg)
        raise ValueError(f"Unknown tokenizer: {name}")

