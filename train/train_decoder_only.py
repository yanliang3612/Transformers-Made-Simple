"""Train decoder_only.py on a tiny character-level language modeling task.

Run:
    python -m train.train_decoder_only
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.decoder_only import DecoderOnlyTransformer
from utils.sampling import generate
from utils.tokenizer import CharTokenizer


def make_lm_batch(data: torch.Tensor, batch_size: int = 16, seq_len: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x, y


def main() -> None:
    torch.manual_seed(0)
    text = (
        "attention looks at useful tokens. "
        "transformers learn patterns from sequences. "
        "small data is enough for a demo. "
    ) * 80
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        max_len=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(50):
        x, y = make_lm_batch(data)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step} decoder_only_loss={loss.item():.4f}")

    prompt = torch.tensor([tokenizer.encode("attention ")], dtype=torch.long)
    sample = generate(model, prompt, max_new_tokens=30, top_k=8)
    print(tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()

