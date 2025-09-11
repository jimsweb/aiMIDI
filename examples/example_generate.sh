#!/usr/bin/env bash
set -euo pipefail

# Example generation
ai_midi generate --prompt examples/sample_prompt.mid --out out.mid --max-tokens 256 --temp 1.0 --top_k 50 --top_p 0.95
