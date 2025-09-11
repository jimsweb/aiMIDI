"""Minimal FastAPI server for MIDI generation.

Exposes /generate endpoint that accepts a prompt MIDI file upload and returns
generated MIDI bytes. Secured with a simple token header `X-API-Token`.
"""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

from ..infer.generator import generate_from_midi


API_TOKEN = "changeme"  # replace via env in production

app = FastAPI(title="midigegen")


def verify_token(x_api_token: Annotated[str | None, Header()] = None) -> None:
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/generate")
def generate(
    file: UploadFile = File(...),
    _: None = Depends(verify_token),
):
    # Save prompt to a temp file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        prompt_path = os.path.join(td, "prompt.mid")
        out_path = os.path.join(td, "out.mid")
        with open(prompt_path, "wb") as f:
            f.write(file.file.read())
        generate_from_midi(prompt_path, out_path, sampling_cfg={"max_tokens": 256})
        with open(out_path, "rb") as f:
            data = f.read()
    return {"midi": data.hex()}  # simple transport; client should hex-decode


if __name__ == "__main__":  # pragma: no cover - manual run
    uvicorn.run(app, host="0.0.0.0", port=8000)

