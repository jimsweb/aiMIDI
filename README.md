# ai_midi

ai_midi is a Python 3.12+ toolkit to train and run generative MIDI models using a REMI-like tokenization, a GPT-style Transformer (PyTorch + PyTorch Lightning), and Hydra configuration. It provides CLI tools to preprocess datasets, train, and generate new MIDI samples, plus an optional FastAPI server.

## Features

- Tokenization: simple REMI-like scheme with combined Note(pitch,duration), Velocity bins, ProgramChange, TimeShift, and special tokens.
- Training: LightningModule Transformer with AMP, checkpointing, cosine/linear LR schedule, gradient clipping.
- Dataset: sliding-window token sequences with on-disk caching.
- Inference: top-k / top-p / temperature sampling, greedy, EOS stopping.
- CLI and optional FastAPI server.
- Hydra configs, structured logging, seeding for reproducibility.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install
```

Note: Installing `torch` may vary by platform/GPU. See https://pytorch.org/get-started/locally/ for the best command for your system.

## Quickstart

1) Put MIDI files in a folder, e.g., `data/midi/`.

2) Preprocess (optional – dataset also processes on the fly and caches):

```bash
ai_midi preprocess --midi-dir data/midi --out-cachedir data/cache
```

3) Train with defaults:

```bash
ai_midi train
```

Or specify config overrides (Hydra):

```bash
ai_midi train train.max_epochs=2 model.d_model=256 data.seq_len=256
```

4) Generate from a short prompt:

```bash
ai_midi generate --prompt examples/sample_prompt.mid --out out.mid --max-tokens 256 --temp 1.0 --top_k 50 --top_p 0.95
```

## Configuration

Configs live under `src/midigegen/config/` and are composed via Hydra. The root `config.yaml` merges `model.yaml`, `data.yaml`, and `train.yaml`. You can override any key from the CLI.

## Dataset Preparation Tips

- Prefer well-quantized MIDI with consistent tempos.
- Remove metadata tracks when not needed; focus on melodic/drum tracks you care about.
- Our tokenizer quantizes to a fixed step; consider setting it to match your musical grid (e.g., 1/32 or 1/64 note).
- Normalize tempos for determinism when training.

## Troubleshooting & Tuning

- If memory is tight, reduce `model.d_model`, `model.n_layers`, `data.seq_len`, and `train.batch_size`.
- Start with greedy or top-k sampling and adjust `temperature` and `top_p` for diversity.
- Warmup steps (`train.warmup_steps`) can stabilize early training.
- Ensure your dataset has sufficient musical variety; augmentation (transpose) may help.

## Testing

Run the tests:

```bash
pytest -q
```

## License

MIT. See `LICENSE`.
