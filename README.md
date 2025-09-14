# ai_midi

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img width="1536" height="1024" alt="ae0905c4-a4c6-4f1a-82be-8cde466723c5" src="https://github.com/user-attachments/assets/cfa74dae-584c-44d4-9699-4ca44835290b" />


`ai_midi` is a Python toolkit for training and generating MIDI with REMI-like tokenization, a GPT-style Transformer powered by PyTorch Lightning, and Hydra-based configuration. It ships with CLI tools to preprocess datasets, train models, sample new music, and even an optional FastAPI server.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Dataset Preparation Tips](#dataset-preparation-tips)
- [Troubleshooting & Tuning](#troubleshooting--tuning)
- [Testing](#testing)
- [License](#license)

## Features

- REMI-inspired tokenization with combined `Note(pitch,duration)` tokens, velocity bins, program changes, time shifts, and special tokens.
- Lightning-powered Transformer training (AMP, checkpointing, cosine/linear LR schedule, gradient clipping).
- Sliding-window dataset with on-disk caching.
- Flexible sampling: top-k, top-p, temperature, greedy, and EOS stopping.
- Command-line tools and optional FastAPI server.
- Hydra configs, structured logging, and reproducible seeding.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pre-commit install
```

See [PyTorch's installation guide](https://pytorch.org/get-started/locally/) for platform-specific `torch` wheels.

## Quickstart

1. Place MIDI files in a folder, e.g. `data/midi/`.
2. *(Optional)* Preprocess and cache sequences:

```bash
ai_midi preprocess --midi-dir data/midi --out-cachedir data/cache
```

3. Train a model:

```bash
ai_midi train
```

   Override any config via Hydra:

```bash
ai_midi train train.max_epochs=2 model.d_model=256 data.seq_len=256
```

4. Generate new music from a prompt:

```bash
ai_midi generate --prompt examples/sample_prompt.mid --out out.mid --max-tokens 256 --temp 1.0 --top_k 50 --top_p 0.95
```

## Configuration

Configurations live in `src/midigegen/config/` and are composed with Hydra. The base `config.yaml` includes `model.yaml`, `data.yaml`, and `train.yaml`. Override any key from the CLI.

## Dataset Preparation Tips

- Use well-quantized MIDI with consistent tempos.
- Remove unneeded metadata tracks; focus on relevant melodic or drum tracks.
- Match the tokenizer's step size to your musical grid (e.g., 1/32 or 1/64).
- Normalize tempos when training.

## Troubleshooting & Tuning

- Reduce `model.d_model`, `model.n_layers`, `data.seq_len`, or `train.batch_size` if memory is tight.
- Start with greedy or top-k sampling, then adjust `temperature` and `top_p` for diversity.
- `train.warmup_steps` can stabilize early training.
- Augment or transpose MIDI for more variety.

## Testing

Run the tests:

```bash
pytest -q
```

## License

Released under the MIT license. See [`LICENSE`](LICENSE) for details.

