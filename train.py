from typing import ClassVar
from dataclasses import dataclass
from enum import Enum, auto
from time import time
import string

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.adamw import AdamW
from torch.optim import Optimizer
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from models.base_language_model import BaseLanguageModel
from models.bigram import BigramLM
from models.recurrent import RecurrentLM, RecurrentLMGraves, RecurrentEnsembleLM


torch.set_float32_matmul_precision('high') # tensor core use


class ModelType(Enum):
    BIGRAM = auto()
    RNN = auto()
    RNNGRAVES = auto()
    RNNENSEMBLE = auto()
    TRANSFORMER = auto()


@dataclass
class TrainConfig:

    # data
    dataset_path: str = "data/tinyshakespeare.txt"
    p_train: float = 0.9

    # training
    epochs = 10
    batch_size: int = 512
    lr: float = 0.003
    clip_grads: float | None = 1.0
    shuffle: bool = True
    compile: bool = True
    mixed_precision: bool = True
    device: str = "cuda"

    # model
    context_length: int = 128
    model: ModelType = ModelType.RNNENSEMBLE
    gen_temperature: float = 0.3


class CharTokenizer:

    _charset: ClassVar[list[str]] = [" ", string.ascii_letters, string.digits, string.punctuation]

    def __init__(self, text: str = "") -> None:
        self._stoi, self._itos = self._build_alphabet(text)

    def _build_alphabet(self, text: str) -> tuple[dict[str, int], dict[int, str]]:
        chars_in_text = set(text)

        alphabet = set()
        for charset in CharTokenizer._charset:
            alphabet.update(charset)
        alphabet.update(chars_in_text)

        vocabulary = sorted(chars_in_text)
        return ({char: i for i, char in enumerate(vocabulary)},
                {i: char for i, char in enumerate(vocabulary)})
    
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
    return TextDataset(train, config.context_length), TextDataset(val, config.context_length)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    *_, C = logits.shape
    logits = logits.view(-1, C)
    targets = targets.view(-1)
    return F.cross_entropy(logits, targets)


def step(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module,
         loss_fn, scaler: GradScaler, optimizer: Optimizer,
         config: TrainConfig) -> tuple[float, float]:

    batch_start = time()
    x, targets = batch
    x, targets = x.to(config.device), targets.to(config.device)

    with torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.mixed_precision):
        logits = model(x)
        loss = loss_fn(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    if config.clip_grads is not None:
        scaler.unscale_(optimizer) # needed for regular clipping
        clip_grad_norm_(model.parameters(), max_norm=config.clip_grads)

    scaler.step(optimizer)
    scaler.update()

    return loss.item(), (time() - batch_start)


def train(model: nn.Module, train_dataset, val_dataset, config: TrainConfig):
    model.train()
    model.to(config.device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=config.shuffle, num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                              shuffle=config.shuffle, num_workers=8,
                              pin_memory=True)
    scaler = torch.amp.GradScaler(device=config.device, enabled=config.mixed_precision)
    optimizer = AdamW(model.parameters(), lr=config.lr)

    for e in range(1, config.epochs+1):
        running_loss = 0.

        for i, batch in enumerate(train_loader):
            loss, batch_time = step(batch, model, loss_fn, scaler, optimizer, config)
            running_loss += loss

            val_result = ""
            if i == len(train_loader) - 1:
                total_val_loss = 0.
                for batch in val_loader:
                    val_loss, _ = step(batch, model, loss_fn, scaler, optimizer, config)
                    total_val_loss += val_loss
                val_result = f", {total_val_loss / len(val_loader):.6f}"
                

            print(f"epoch {e}, batch {i}/{len(train_loader)-1}, loss {running_loss/(i+1):.6f}" +
                  val_result + f", s/it {batch_time:.4f}")


def sample_text(model: BaseLanguageModel, tokenizer: CharTokenizer, max_new_tokens: int, config: TrainConfig):
    model.to(config.device)
    model.eval()
    start_tokens = torch.zeros(size=(1,1), dtype=torch.long, device=config.device)
    preds = model.generate(start_tokens, max_new_tokens)
    for pred in preds:
        sample_text = tokenizer.decode(pred[1:].tolist())
        print(sample_text + "\n")


def build_model(vocab_size: int, config: TrainConfig):

    match config.model:
        case ModelType.BIGRAM:
            return BigramLM(vocab_size)
        case ModelType.RNN:
            # Turns out multiple stacked layers of this model perform poorly, probably a lot
            # of the input information gets lost passing from layer to layer.
            return RecurrentLM(vocab_size, hidden_dim=256, num_layers=5)
        case ModelType.RNNGRAVES:
            return RecurrentLMGraves(vocab_size, hidden_dim=256, num_layers=5)
        case ModelType.RNNENSEMBLE:
            return RecurrentEnsembleLM(vocab_size, embed_dim=32, hidden_dim=256, num_layers=5)
        
    raise KeyError("Specified model type {config.model} not available.")


def main():
    config = TrainConfig()
    text = load_text(config.dataset_path)
    tokenizer = CharTokenizer(text)

    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = create_datasets(encoded_text, config)
    
    model = build_model(len(tokenizer), config)
    if config.compile:
        model.compile() # issues with shape transforms in ModelType.RNNGRAVES on 'cpu'

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    train(model, train_dataset, val_dataset, config)
    sample_text(model, tokenizer, max_new_tokens=500, config=config)


if __name__ == "__main__":
    main()