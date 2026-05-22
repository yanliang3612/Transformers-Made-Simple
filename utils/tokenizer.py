"""A beginner-friendly character tokenizer.

Real projects usually use BPE/SentencePiece tokenizers. A character tokenizer is
useful here because it has no external files and makes examples easy to run.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, text: str, pad_token: str = "<pad>", unk_token: str = "<unk>") -> None:
        chars = sorted(set(text))
        self.itos = [pad_token, unk_token] + chars
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_id = self.stoi[pad_token]
        self.unk_id = self.stoi[unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i < len(self.itos))

