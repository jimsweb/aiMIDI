"""Command-line interface for ai_midi.

Provides subcommands for training, generation, preprocessing, and evaluation.
Uses Hydra for configuration overrides and argparse for the entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from .train.trainer import run_train
from .infer.generator import generate_from_midi
from .utils.logging import configure_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai_midi", description="MIDI generative modeling CLI")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    train_p = sub.add_parser("train", help="Train model (Hydra overrides allowed)")
    train_p.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides a=b c=d ...")
    train_p.add_argument("--dry-run", action="store_true", help="Initialize then exit")
    train_p.add_argument("--debug", action="store_true", help="More verbose logging")

    # Generate
    gen_p = sub.add_parser("generate", help="Generate from a prompt MIDI")
    gen_p.add_argument("--prompt", required=True, help="Path to prompt MIDI file")
    gen_p.add_argument("--out", required=True, help="Path to save generated MIDI file")
    gen_p.add_argument("--checkpoint", required=False, help="Model checkpoint path")
    gen_p.add_argument("--max-tokens", type=int, default=512)
    gen_p.add_argument("--temp", type=float, default=1.0)
    gen_p.add_argument("--top_k", type=int, default=50)
    gen_p.add_argument("--top_p", type=float, default=0.95)
    gen_p.add_argument("--debug", action="store_true")

    # Preprocess
    prep_p = sub.add_parser("preprocess", help="Preprocess MIDI dir and cache tokens")
    prep_p.add_argument("--midi-dir", required=True)
    prep_p.add_argument("--out-cachedir", required=True)
    prep_p.add_argument("--reprocess", action="store_true")
    prep_p.add_argument("--debug", action="store_true")

    # Evaluate (placeholder)
    eval_p = sub.add_parser("evaluate", help="Evaluate model on a dataset")
    eval_p.add_argument("--model-checkpoint", required=True)
    eval_p.add_argument("--dataset", required=True)
    eval_p.add_argument("--debug", action="store_true")

    return parser.parse_args()


def _hydra_main(cfg: DictConfig, dry_run: bool = False) -> None:
    if dry_run:
        print(OmegaConf.to_yaml(cfg))
        return
    run_train(cfg)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args()
    level = "DEBUG" if getattr(args, "debug", False) else "INFO"
    configure_logging(level)

    if args.cmd == "train":
        overrides: list[str] = args.overrides or []

        @hydra.main(version_base=None, config_path="config", config_name="config")
        def _entry(cfg: DictConfig) -> None:  # type: ignore[misc]
            _hydra_main(cfg, dry_run=args.dry_run)

        sys.argv = [sys.argv[0]] + overrides
        _entry()  # type: ignore[misc]

    elif args.cmd == "generate":
        sampling_cfg: Dict[str, Any] = {
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "checkpoint": args.checkpoint,
        }
        generate_from_midi(
            prompt_mid_path=args.prompt,
            out_mid_path=args.out,
            sampling_cfg=sampling_cfg,
        )

    elif args.cmd == "preprocess":
        # Dataset builds cache when first accessed; here we force the pass.
        from .data.dataset import MIDIDataset
        from .data.tokenizer import TokenizerFactory
        from omegaconf import OmegaConf

        tok = TokenizerFactory.get("remi")
        ds = MIDIDataset(
            midi_dir=args.midi_dir,
            tokenizer=tok,
            cache_dir=args.out_cachedir,
            reprocess=args.reprocess,
        )
        print(f"Cached {len(ds._all_tokens)} token sequences to {args.out_cachedir}")

    elif args.cmd == "evaluate":
        print("Evaluation is not yet implemented. Placeholder.")

    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
