from typing import Final
from dataclasses import dataclass
import string

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


_CHARSETS: Final = [" ", string.ascii_letters, string.digits, string.punctuation]


@dataclass
class TrainConfig:
    dataset_path: str = "data/tinyshakespeare.txt"
    p_train: float = 0.9
    batch_size: int = 4
    block_size: int = 8 # max context length


class CharTokenizer:

    def __init__(self, text) -> None:
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


def main():
    config = TrainConfig()
    text = load_text(config.dataset_path)
    tokenizer = CharTokenizer(text)

    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = create_datasets(encoded_text, config)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    for i, (x, y) in enumerate(train_dataloader):
        if i == 3:
            break
        print(x.shape, y.shape)



if __name__ == "__main__":
    main()