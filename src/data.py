"""
Data and tokenization.

We work at the CHARACTER level: each character of the text becomes an integer.
It is the simplest choice there is, and that is deliberate: we want all the
project's complexity to live in the model, not in the tokenizer.

(Real LLMs use BPE, which splits into subwords. We could add it later as an
extension; it is not the point here.)
"""

from pathlib import Path
import torch


class CharTokenizer:
    """Maps each unique character of the corpus to an integer, and back."""

    def __init__(self, text: str):
        # sorted() so the mapping is deterministic from one run to the next.
        # Without it the order would change and a saved checkpoint would become
        # unreadable.
        chars = sorted(set(text))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int -> string

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_data(path: str | Path, train_frac: float = 0.9, device: str = "cpu"):
    """
    Read the text, tokenize it, and split into train/validation.

    SEQUENTIAL split (first 90% for training, last 10% for validation), not
    random. On text, a random split would leak context: the model would see
    passages adjacent to validation ones during training, and the validation
    score would be too optimistic.
    """
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer(text)

    # dtype=long because the indices are used to index an embedding table.
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n = int(train_frac * len(data))
    train_data = data[:n].to(device)
    val_data = data[n:].to(device)
    return tok, train_data, val_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = "cpu"):
    """
    Draw a random batch of examples.

    block_size = context length: how many characters the model sees before
    predicting the next one.

    The key point is the offset of 1 between x and y:
        text  : "hello"
        x     : "hell"
        y     : "ello"
    For each position t, the model sees x[:t+1] and must predict y[t]. A single
    example of block_size characters thus provides block_size predictions to
    supervise, not just one. That is what makes training efficient.
    """
    # One random start position per batch element.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
