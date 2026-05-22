"""Train a tiny GPT-like model on a short string.

Run:
    python -m train.train_lm
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from configs.gpt_config import GPTConfig
from models.decoder_only import DecoderOnlyTransformer
from utils.sampling import generate
from utils.tokenizer import CharTokenizer


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x, y


def main() -> None:
    text = "transformers made simple. attention learns which tokens matter. " * 100
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, dim=128, num_layers=2, num_heads=4, mlp_dim=512)
    model = DecoderOnlyTransformer(
        cfg.vocab_size,
        cfg.dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.mlp_dim,
        cfg.max_len,
        cfg.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(50):
        x, y = make_batch(data, batch_size=16, seq_len=32)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    prompt = torch.tensor([tokenizer.encode("transformers ")], dtype=torch.long)
    generated = generate(model, prompt, max_new_tokens=40, top_k=10)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
