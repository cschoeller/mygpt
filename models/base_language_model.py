from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseLanguageModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, token_indices: torch.Tensor):
        ...
    
    @abstractmethod
    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        ...
