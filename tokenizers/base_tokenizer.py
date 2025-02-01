from abc import ABC, abstractmethod

class BaseTokenizer(ABC):

    @abstractmethod
    def train(self, text: str):
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...