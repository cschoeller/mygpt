from dataclasses import dataclass, field
from enum import Enum, auto
import os
import pickle
from time import time

import torch
from torch import nn
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from models.base_language_model import BaseLanguageModel
from models.bigram import BigramLM
from models.gpt import KarpathyGPT
from models.recurrent import RecurrentEnsembleLM, RecurrentLM, RecurrentLMGraves
from models.transformer import Transformer, TransformerParams
from models.transformer_vanilla import TransformerVanilla
from tokenizers.byte_pair_tokenizer import BytePairTokenizer
from tokenizers.char_tokenizer import CharTokenizer

torch.set_float32_matmul_precision("high")  # enable tensor cores

if torch.backends.cuda.is_flash_attention_available():
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)


class ModelType(Enum):
    BIGRAM = auto()
    RNN = auto()
    RNNGRAVES = auto()
    RNNENSEMBLE = auto()
    TRANSFORMERVANILLA = auto()
    TRANSFORMER = auto()
    KARPATHY = auto()


@dataclass
class TrainConfig:
    # data
    dataset_path: str = "data/tinyshakespeare.txt"  # "data/BNCSplitWordsCorpus.txt"
    p_train: float = 0.9

    # training
    epochs = 1
    batch_size: int = 64
    lr: float = 1e-3
    clip_grads: float | None = 1
    shuffle: bool = True
    compile: bool = True
    mixed_precision: bool = True
    device: str = "cuda"

    # regularization
    weight_decay: float = 0.01

    # model
    model: ModelType = ModelType.TRANSFORMER
    gen_temperature: float = 1.0
    transformer_params: TransformerParams = field(
        default_factory=lambda: TransformerParams(
            vocab_size=512,
            context_length=256,
            embed_dim=384,
            heads=6,
            n_layers=6,
            drop_rate=0.2,
            ffn_activation="relu",  # {"relu", "gelu", "relu2"}
            normalization="layernorm",  # {"layernorm", "dynamic_tanh"}
            positional_encoding="learned",  # {"learned", "sinusoidal", "nope"}
        )
    )


class TextDataset(Dataset):
    def __init__(self, text_tensor: torch.Tensor, block_size: int) -> None:
        self._text_tensor = text_tensor
        self._block_size = block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._text_tensor[idx : idx + self._block_size]
        y = self._text_tensor[idx + 1 : idx + self._block_size + 1]
        return x, y

    def __len__(self) -> int:
        return len(self._text_tensor) - self._block_size


def load_text(path_to_textfile: str) -> str:
    with open(path_to_textfile, "r") as f:
        return f.read()


def create_datasets(encoded_text: torch.Tensor, config: TrainConfig) -> tuple[TextDataset, TextDataset]:
    n = int(config.p_train * len(encoded_text))
    train, val = encoded_text[:n], encoded_text[n:]
    context_length = config.transformer_params.context_length
    return TextDataset(train, context_length), TextDataset(val, context_length)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    *_, C = logits.shape
    logits = logits.view(-1, C)
    targets = targets.view(-1)
    return F.cross_entropy(logits, targets)


def step(
    batch: tuple[torch.Tensor, torch.Tensor],
    model: nn.Module,
    loss_fn,
    scaler: GradScaler,
    optimizer: Optimizer,
    config: TrainConfig,
    is_train: bool = True,
) -> tuple[float, float]:
    batch_start = time()
    x, targets = batch
    x, targets = x.to(config.device), targets.to(config.device)

    # some speedup from optimizer compilation, reduce-overhead (cuda graphs)
    # and disabling grad in validation
    @torch.compile(disable=not config.compile, mode="reduce-overhead")
    def _compiled_step(is_train: bool) -> torch.Tensor:
        with (
            torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.mixed_precision),
            torch.set_grad_enabled(is_train),
        ):
            logits = model(x)
            loss = loss_fn(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if config.clip_grads is not None:
                scaler.unscale_(optimizer)  # needed for regular clipping
                clip_grad_norm_(model.parameters(), max_norm=config.clip_grads)

            scaler.step(optimizer)
            scaler.update()

        return loss

    loss = _compiled_step(is_train)
    return loss.item(), (time() - batch_start)


def train(model: nn.Module, train_dataset, val_dataset, config: TrainConfig):
    model.train()
    model.to(config.device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8, pin_memory=True
    )
    scaler = torch.amp.GradScaler(device=config.device, enabled=config.mixed_precision)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for e in range(1, config.epochs + 1):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            loss, batch_time = step(batch, model, loss_fn, scaler, optimizer, config)
            running_loss += loss

            val_result = ""
            if i == len(train_loader) - 1:
                total_val_loss = 0.0
                for batch in val_loader:
                    val_loss, _ = step(batch, model, loss_fn, scaler, optimizer, config, is_train=False)
                    total_val_loss += val_loss
                val_result = f", val_loss {total_val_loss / len(val_loader):.6f}"

            print(
                f"epoch {e}, batch {i}/{len(train_loader) - 1}, loss {running_loss / (i + 1):.6f}"
                + val_result
                + f", s/it {batch_time:.4f}"
            )


def sample_text(model: BaseLanguageModel, tokenizer: CharTokenizer, max_new_tokens: int, config: TrainConfig):
    model.to(config.device)
    model.eval()
    start_tokens = torch.zeros(size=(1, 1), dtype=torch.long, device=config.device)
    preds = model.generate(start_tokens, max_new_tokens, config.gen_temperature)
    for pred in preds:
        sample_text = tokenizer.decode(pred[1:].tolist())
        print(sample_text + "\n")


def build_model(config: TrainConfig):
    transformer_params = config.transformer_params
    match config.model:
        case ModelType.BIGRAM:
            return BigramLM(transformer_params.vocab_size)
        case ModelType.RNN:
            # Turns out multiple stacked layers of this model perform poorly, probably a lot
            # of the input information gets lost passing from layer to layer.
            return RecurrentLM(transformer_params.vocab_size, hidden_dim=256, num_layers=5)
        case ModelType.RNNGRAVES:
            return RecurrentLMGraves(transformer_params.vocab_size, hidden_dim=256, num_layers=5)
        case ModelType.RNNENSEMBLE:
            return RecurrentEnsembleLM(transformer_params.vocab_size, embed_dim=32, hidden_dim=256, num_layers=5)
        case ModelType.TRANSFORMERVANILLA:
            return TransformerVanilla(
                transformer_params.vocab_size,
                context_length=transformer_params.context_length,
                embed_dim=transformer_params.embed_dim,
                heads=transformer_params.heads,
                n_layers=transformer_params.n_layers,
                drop_rate=transformer_params.drop_rate,
            )
        case ModelType.TRANSFORMER:
            return Transformer(transformer_params)
        case ModelType.KARPATHY:
            return KarpathyGPT()

    raise KeyError("Specified model type {config.model} not available.")


def main():
    config = TrainConfig()
    text = load_text(config.dataset_path)

    # tokenizer = CharTokenizer()
    if os.path.exists("bpe_tokenizer.pkl"):
        print("Loading from tokenizer from disk...")
        tokenizer = pickle.load(open("bpe_tokenizer.pkl", "rb"))
    else:
        print("Training tokenizer...")
        tokenizer = BytePairTokenizer(vocab_size=config.transformer_params.vocab_size)
        tokenizer.train(text)
        pickle.dump(tokenizer, open("bpe_tokenizer.pkl", "wb"))

    print("Encoding training text...")
    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = create_datasets(encoded_text, config)

    print("Building model...")
    model = build_model(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    print("Training model...")
    train(model, train_dataset, val_dataset, config)

    print("Saving checkpoint...")
    torch.save(model.state_dict(), f"{config.model.name.lower()}_checkpoint_{config.epochs}ep.pt")
    # model.load_state_dict(torch.load("transformer_checkpoint_5ep.pt", weights_only=True))

    print("Generating text...")
    sample_text(model, tokenizer, max_new_tokens=1000, config=config)


if __name__ == "__main__":
    main()
