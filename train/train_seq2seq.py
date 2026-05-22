"""Train a tiny encoder-decoder Transformer on a toy copy task."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from configs.t5_config import T5Config
from models.encoder_decoder import EncoderDecoderTransformer
from modules.masks import padding_mask


def main() -> None:
    cfg = T5Config(src_vocab_size=100, tgt_vocab_size=100, dim=128, num_layers=2, num_heads=4, mlp_dim=512)
    model = EncoderDecoderTransformer(
        cfg.src_vocab_size,
        cfg.tgt_vocab_size,
        cfg.dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.mlp_dim,
        cfg.max_len,
        cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(30):
        src = torch.randint(1, cfg.src_vocab_size, (16, 16))
        target = src.clone()
        decoder_in = torch.cat([torch.ones(16, 1, dtype=torch.long), target[:, :-1]], dim=1)

        logits = model(src, decoder_in, src_mask=padding_mask(src, cfg.pad_id))
        loss = F.cross_entropy(logits.view(-1, cfg.tgt_vocab_size), target.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} seq2seq_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
