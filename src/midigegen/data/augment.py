"""Simple data augmentation utilities.

Currently provides transposition. Time-stretch is left as a placeholder
because tempo/time grid is quantized at tokenization.
"""

from __future__ import annotations

from typing import Iterable, List


def transpose_token_sequence(tokens: List[int], pitch_change: int, id_to_event: dict[int, str], event_to_id: dict[str, int]) -> List[int]:
    """Transpose pitch in Note_<p>_Dur_<d> tokens by `pitch_change` semitones.

    Tokens outside 0..127 after transpose are clipped.
    """

    out: List[int] = []
    for t in tokens:
        s = id_to_event.get(int(t), "")
        if s.startswith("Note_") and "_Dur_" in s:
            _, rest = s.split("Note_")
            p_str, d_str = rest.split("_Dur_")
            p = max(0, min(127, int(p_str) + pitch_change))
            d = int(d_str)
            out.append(event_to_id.get(f"Note_{p}_Dur_{d}", t))
        else:
            out.append(t)
    return out

