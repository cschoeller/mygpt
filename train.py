from dataclasses import dataclass, field
import datetime
from enum import Enum, auto
import os
import pickle
from time import time
from typing import Final

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch import nn
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset

from models.base_language_model import BaseLanguageModel
from models.bigram import BigramLM
from models.gpt import KarpathyGPT
from models.recurrent import RecurrentEnsembleLM, RecurrentLM, RecurrentLMGraves
from models.transformer import Transformer, TransformerParams
from models.transformer_vanilla import TransformerVanilla
from optimizer.muon import Muon, split_params_for_muon
from tokenizers.base_tokenizer import BaseTokenizer
from tokenizers.byte_pair_tokenizer import BytePairTokenizer
from tokenizers.char_tokenizer import CharTokenizer

torch.set_float32_matmul_precision("high")  # enable tensor cores

if torch.backends.cuda.is_flash_attention_available():
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

# TODO:
# - Finish integrating tiktoken gpt-2
# - Don't save all intermediate outputs with skip connections in transformer
# - Shuffle dataset chunks before splitting to train and val to avoid bias


_CONTEXT_LENGTH: Final = 256


class TokenizerType(Enum):
    CUSTOM_BPE = auto()
    CHAR = auto()
    TIKTOKEN = auto()


class OptimizerType(Enum):
    ADAMW = auto()
    MUON = auto()
    ADAMUON = auto()


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
    dataset_paths: str = field(
        default_factory=lambda: [
            # "data/tinyshakespeare.txt",
            "data/BNCSplitWordsCorpus.txt",
            # "data/taylorswift.txt",
        ]
    )
    text_stride: int | None = _CONTEXT_LENGTH // 4  # None: defaults to context_length
    p_train: float = 0.9

    # training
    epochs = 20
    batch_size: int = 128
    lr: float = 1e-3
    lr_muon: float = 1e-2
    use_scheduler: bool = True
    clip_grads: float | None = 1.0
    shuffle: bool = True
    compile: bool = True
    mixed_precision: bool = True
    device: str = "cuda"
    optimizer: OptimizerType = OptimizerType.ADAMUON

    # regularization
    weight_decay: float = 0.01

    # model
    model: ModelType = ModelType.TRANSFORMER
    gen_temperature: float = 1.0
    transformer_params: TransformerParams = field(
        default_factory=lambda: TransformerParams(
            vocab_size=512,
            context_length=_CONTEXT_LENGTH,
            embed_dim=256,  # 384
            heads=4,  # 6
            n_layers=3,  # 6
            drop_rate=0.2,
            u_net_skips=False,
            ffn_activation="relu",  # {"relu", "gelu", "relu2"}
            normalization="layernorm",  # {"layernorm", "dynamic_tanh"}
            positional_encoding="rotary",  # {"learned", "sinusoidal", "nope", "rotary"}
            qk_norm=True,
        )
    )
    tokenizer = TokenizerType.CUSTOM_BPE


class TextDataset(Dataset):
    def __init__(self, text_tensor: torch.Tensor, block_size: int, stride: int = 1) -> None:
        self._text_tensor = text_tensor
        self._block_size = block_size
        self._stride = stride

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * self._stride
        x = self._text_tensor[offset : offset + self._block_size]
        y = self._text_tensor[offset + 1 : offset + self._block_size + 1]
        return x, y

    def __len__(self) -> int:
        return (len(self._text_tensor) - self._block_size) // self._stride


def load_text(path_to_textfile: str) -> str:
    with open(path_to_textfile, "r") as f:
        return f.read()


def build_datasets(encoded_texts: list[torch.Tensor], config: TrainConfig) -> tuple[Dataset, Dataset]:
    train_datasets = []
    val_datasets = []
    stride = config.transformer_params.context_length if config.text_stride is None else config.text_stride
    for encoded_text in encoded_texts:
        n = int(config.p_train * len(encoded_text))
        train, val = encoded_text[:n], encoded_text[n:]
        context_length = config.transformer_params.context_length
        train_datasets.append(TextDataset(train, context_length, stride=stride))
        val_datasets.append(TextDataset(val, context_length, stride=stride))
    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


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
    optimizers: list[Optimizer],
    config: TrainConfig,
    is_train: bool = True,
    schedulers: list[SequentialLR] | None = None,
) -> tuple[float, float]:
    batch_start = time()
    x, targets = batch
    x, targets = x.to(config.device), targets.to(config.device)

    with (
        torch.autocast(device_type=config.device, dtype=torch.float16, enabled=config.mixed_precision),
        torch.set_grad_enabled(is_train),
    ):
        torch.compiler.cudagraph_mark_step_begin()
        logits = model(x)
        loss = loss_fn(logits, targets)

    if is_train:
        for optim in optimizers:
            optim.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()

        if config.clip_grads is not None:
            params = {}
            for optim in optimizers:
                scaler.unscale_(optim)  # needed for regular clipping
                for group in optim.param_groups:
                    for p in group["params"]:
                        params[id(p)] = p
            clip_grad_norm_(params.values(), max_norm=config.clip_grads)

        # step optimizer and corresponding scheduler
        for i, optim in enumerate(optimizers):
            scaler.step(optim)
            if schedulers is not None:
                schedulers[i].step()

        scaler.update()

    return loss.item(), (time() - batch_start)


def train(model: nn.Module, train_dataset, val_dataset, config: TrainConfig) -> tuple[list[float], list[float]]:
    model.train()
    model.to(config.device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8, pin_memory=True
    )
    scaler = torch.amp.GradScaler(device=config.device, enabled=config.mixed_precision)
    optimizers = build_optimizers(model, config)
    schedulers = build_schedulers(optimizers, dataset_size=len(train_loader), config=config)

    # compile model and loss for speedup, but not the optimizer loop, as this causes issues with Muon
    model_compiled = torch.compile(model, disable=not config.compile, mode="reduce-overhead")
    loss_fn_compiled = torch.compile(loss_fn, disable=not config.compile, mode="reduce-overhead")

    train_losses = []
    val_losses = []

    for e in range(1, config.epochs + 1):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            loss, batch_time = step(
                batch, model_compiled, loss_fn_compiled, scaler, optimizers, config, schedulers=schedulers
            )
            running_loss += loss

            val_result = ""
            if i == len(train_loader) - 1:
                total_val_loss = 0.0
                for batch in val_loader:
                    val_loss, _ = step(
                        batch, model_compiled, loss_fn_compiled, scaler, optimizers, config, is_train=False
                    )
                    total_val_loss += val_loss

                avg_val_loss = total_val_loss / len(val_loader)
                val_result = f", val_loss {avg_val_loss:.6f}"

                val_losses.append(avg_val_loss)
                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)

            print(
                f"epoch {e}, batch {i}/{len(train_loader) - 1}, loss {running_loss / (i + 1):.6f}"
                + val_result
                + f", s/it {batch_time:.4f}"
            )

    return train_losses, val_losses


def plot_loss_curves(train_losses: list[float], val_losses: list[float], out_path: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train loss", marker="o")
    plt.plot(epochs, val_losses, label="val loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curve to {out_path}")


def sample_text(model: BaseLanguageModel, tokenizer: CharTokenizer, max_new_tokens: int, config: TrainConfig):
    model.to(config.device)
    model.eval()
    start_tokens = torch.zeros(size=(1, 1), dtype=torch.long, device=config.device)
    preds = model.generate(start_tokens, max_new_tokens, config.gen_temperature)
    for pred in preds:
        sample_text = tokenizer.decode(pred[1:].tolist())
        print(sample_text + "\n")


def build_optimizers(model: nn.Module, config: TrainConfig) -> list[Optimizer]:
    match config.optimizer:
        case OptimizerType.ADAMW:
            return [AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, foreach=True)]
        case OptimizerType.MUON:
            return [Muon(model.parameters(), lr=config.lr_muon, weight_decay=config.weight_decay, nesterov=True)]
        case OptimizerType.ADAMUON:
            muon_params, adamw_params = split_params_for_muon(model)
            muon = Muon(muon_params, lr=config.lr_muon, weight_decay=config.weight_decay, nesterov=True)
            adamw = AdamW(
                adamw_params, lr=config.lr, weight_decay=0.0, foreach=True
            )  # don't decay embeddings, biases, inputs, outputs
            return [muon, adamw]
        case _:
            raise KeyError(f"Specified optimizer type {config.optimizer} not available.")


def build_schedulers(optimizers: list[Optimizer], dataset_size: int, config: TrainConfig) -> list[SequentialLR] | None:
    if not config.use_scheduler:
        return None

    total_iters = config.epochs * dataset_size
    warmup_iters = int(total_iters * 0.1)

    schedulers = []
    for optim in optimizers:
        warmup = LinearLR(optim, start_factor=0.01, total_iters=warmup_iters)
        cosine = CosineAnnealingLR(optim, T_max=(total_iters - warmup_iters))
        combined_scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_iters])
        schedulers.append(combined_scheduler)

    return schedulers


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


def build_tokenizer(texts: list[str], config: TrainConfig) -> BaseTokenizer | tiktoken.Encoding:

    match config.tokenizer:
        case TokenizerType.CHAR:
            print("Using character level tokenizer....")
            return CharTokenizer()
        case TokenizerType.CUSTOM_BPE:
            if os.path.exists("bpe_tokenizer.pkl"):
                print("Loading custom from tokenizer from disk...")
                return pickle.load(open("bpe_tokenizer.pkl", "rb"))
            else:
                print("Training custom tokenizer...")
                tokenizer = BytePairTokenizer(vocab_size=config.transformer_params.vocab_size)
                tokenizer.train("\n".join(texts))
                pickle.dump(tokenizer, open("bpe_tokenizer.pkl", "wb"))
                return tokenizer
        case TokenizerType.TIKTOKEN:
            enc = tiktoken.encoding_for_model("gpt2")
            config.transformer_params.vocab_size = enc.n_vocab
            print("Using tiktoken gpt-2 tokenizer with vocab size", enc.n_vocab)
            return enc
        case _:
            raise KeyError(f"Specified tokenizer type {config.tokenizer} not available.")


def main():
    experiment_start = time()
    config = TrainConfig()
    texts = [load_text(dataset_path) for dataset_path in config.dataset_paths]

    print("Building tokenizer...")
    tokenizer = build_tokenizer(texts, config)

    print("Encoding training text...")
    encoded_texts = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in texts]
    train_dataset, val_dataset = build_datasets(encoded_texts, config)

    print("Building model...")
    model = build_model(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    print("Training model...")
    train_losses, val_losses = train(model, train_dataset, val_dataset, config)

    print("Plotting loss curves...")
    plot_loss_curves(
        train_losses,
        val_losses,
        out_path=f"{config.model.name.lower()}_loss_{config.epochs}ep.png",
    )

    print("Saving checkpoint...")
    torch.save(model.state_dict(), f"{config.model.name.lower()}_checkpoint_{config.epochs}ep.pt")
    # model.load_state_dict(torch.load("transformer_checkpoint_5ep.pt", weights_only=True))

    print("Generating text...")
    sample_text(model, tokenizer, max_new_tokens=1000, config=config)

    print("Perplexity score:", torch.exp(torch.tensor(val_losses[-1])))
    print("Total experiment time:", datetime.timedelta(seconds=int(time() - experiment_start)))


if __name__ == "__main__":
    main()
