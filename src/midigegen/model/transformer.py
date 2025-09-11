"""Decoder-only Transformer LightningModule for language modeling over tokens.

Implements a GPT-like model with positional embeddings, multi-head attention,
and a causal mask. Provides training step, validation step, and a generate
method used by inference utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class GPTModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pad_id = pad_id

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        def lr_lambda(step: int) -> float:
            if self.hparams.warmup_steps <= 0:
                return 1.0
            return min(1.0, (step + 1) / float(self.hparams.warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        device = input_ids.device
        pos = torch.arange(0, s, device=device, dtype=torch.long).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        mask = self._causal_mask(s, device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.pad_id)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        logits = self(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.pad_id)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids.clone()
        device = input_ids.device
        for _ in range(max_new_tokens):
            x = out[:, -self.seq_len :]
            logits = self(x)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)
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
                next_id = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, 1, next_id)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
            if eos_id is not None and (next_token == eos_id).any():
                break
        return out

