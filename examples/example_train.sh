#!/usr/bin/env bash
set -euo pipefail

# Example training command with overrides
ai_midi train train.max_epochs=2 model.d_model=256 model.n_layers=4 data.seq_len=256
