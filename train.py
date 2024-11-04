from typing import Final
from dataclasses import dataclass
import string
from enum import Enum, auto

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adamw import AdamW

from models import BigramLanguageModel


_CHARSETS: Final = [" ", string.ascii_letters, string.digits, string.punctuation]


class ModelType(Enum):
    BIGRAM = auto()


@dataclass
class TrainConfig:
    dataset_path: str = "data/tinyshakespeare.txt"
    p_train: float = 0.9
    epochs = 10
    batch_size: int = 4
    lr: float = 0.01
    shuffle: bool = True
    block_size: int = 8 # max context length
    model: ModelType = ModelType.BIGRAM


class CharTokenizer:

    def __init__(self, text: str = "") -> None:
        self._stoi, self._itos = self._build_alphabet(text)

    def _build_alphabet(self, text: str) -> tuple[dict[str, int], dict[int, str]]:
        chars_in_text = set(text)

        alphabet = set()
        for charset in _CHARSETS:
            alphabet.update(charset)
        alphabet.update(chars_in_text)

        return ({char: i for i, char in enumerate(sorted(alphabet))},
                {i: char for i, char in enumerate(sorted(alphabet))})
    
    def encode(self, text: str) -> list[int]:
        return [self._stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self._itos[id] for id in tokens])
    
    def __len__(self) -> int:
        """Returns the vocabulary size."""
        return len(self._stoi)


class TextDataset(Dataset):
    
    def __init__(self, text_tensor: torch.Tensor, block_size: int) -> None:
        self._text_tensor = text_tensor
        self._block_size = block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._text_tensor[idx: idx + self._block_size]
        y = self._text_tensor[idx+1 : idx + self._block_size + 1]
        return x, y
    
    def __len__(self) -> int:
        return len(self._text_tensor) - self._block_size


def load_text(path_to_textfile: str) -> str:
    with open(path_to_textfile, "r") as f:
        return f.read()


def create_datasets(encoded_text: torch.Tensor, config: TrainConfig) -> tuple[TextDataset, TextDataset]:
    n = int(config.p_train * len(encoded_text))
    train, val = encoded_text[:n], encoded_text[n:]
    return TextDataset(train, config.block_size), TextDataset(val, config.block_size)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, C = logits.shape
    logits = logits.view(-1, C)
    targets = targets.view(-1)
    return F.cross_entropy(logits, targets)


def train(model: nn.Module, train_dataset, val_dataset, config: TrainConfig):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    for e in range(config.epochs):
        running_loss = 0.
        for i, batch in enumerate(train_loader):
            x, targets = batch
            logits = model(x)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            print(f"epoch {e} batch {i}/{len(train_loader)} loss {running_loss/(i+1)}")


def main():
    config = TrainConfig()
    text = load_text(config.dataset_path)
    tokenizer = CharTokenizer(text)

    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = create_datasets(encoded_text, config)
    
    model = BigramLanguageModel(len(tokenizer))
    train(model, train_dataset, val_dataset, config)


if __name__ == "__main__":
    main()