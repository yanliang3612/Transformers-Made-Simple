"""Train encoder_decoder.py on a tiny sequence copy/reverse task.

Run:
    python -m train.train_encoder_decoder
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.encoder_decoder import EncoderDecoderTransformer
from modules.masks import padding_mask


def make_seq2seq_batch(
    batch_size: int = 16,
    seq_len: int = 10,
    vocab_size: int = 80,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create source tokens and target tokens.

    Target rule:
        reverse the source sequence. This is a tiny translation-like task.
    """

    src = torch.randint(3, vocab_size, (batch_size, seq_len))
    target = torch.flip(src, dims=[1])
    bos = torch.ones(batch_size, 1, dtype=torch.long)
    decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
    return src, decoder_input, target


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 80
    model = EncoderDecoderTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_len=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(50):
        src, decoder_input, target = make_seq2seq_batch(vocab_size=vocab_size)
        logits = model(src, decoder_input, src_mask=padding_mask(src, pad_id=0))
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} encoder_decoder_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

