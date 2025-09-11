"""Generation utilities and CLI integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..data.midi_io import save_midi, load_midi
from ..data.tokenizer import RemiTokenizer, TokenizerFactory
from ..model.transformer import GPTModel


def top_k_top_p_filtering(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if top_k is not None:
        v, _ = torch.topk(probs, k=top_k)
        minv = v[:, -1].unsqueeze(-1)
        probs = torch.where(probs < minv, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        # Project back to original order
        probs = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs)
    return probs


@torch.no_grad()
def sample_and_save(model: GPTModel, out_path: str, max_tokens: int = 128) -> None:
    tok = TokenizerFactory.get("remi")
    sos_id = 1  # from SPECIAL order [PAD=0, SOS=1, EOS=2]
    x = torch.tensor([[sos_id]], device=model.device)
    ids = model.generate(x, max_new_tokens=max_tokens, eos_id=2)
    midi = tok.decode(ids[0].tolist())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_midi(midi, out_path)


def generate_from_midi(prompt_mid_path: str, out_mid_path: str, sampling_cfg: Dict[str, Any]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = TokenizerFactory.get("remi")
    sos_id = 1
    eos_id = 2

    prompt_midi = load_midi(prompt_mid_path)
    prompt_ids = tok.encode(prompt_midi)
    x = torch.tensor([prompt_ids[: tok.cfg.max_duration_steps] if prompt_ids else [sos_id]], device=device)

    vocab_size = tok.vocab_size
    model = GPTModel(vocab_size=vocab_size)
    ckpt = sampling_cfg.get("checkpoint")
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    out_ids = model.generate(
        x,
        max_new_tokens=int(sampling_cfg.get("max_tokens", 512)),
        temperature=float(sampling_cfg.get("temperature", 1.0)),
        top_k=sampling_cfg.get("top_k"),
        top_p=sampling_cfg.get("top_p"),
        eos_id=eos_id,
    )
    midi = tok.decode(out_ids[0].tolist())
    save_midi(midi, out_mid_path)

