from typing import ClassVar
import string
from tokenizers.base_tokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):

    _charset: ClassVar[list[str]] = [" ", string.ascii_letters, string.digits, string.punctuation]

    def __init__(self, complete_vocabulary: bool = False) -> None:
        self._complete_vocabulary = complete_vocabulary
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {} #self._build_alphabet(text)

    def train(self, text: str) -> None:
        self._stoi, self._itos = self._build_alphabet(text)

    def _build_alphabet(self, text: str) -> tuple[dict[str, int], dict[int, str]]:
        vocabulary = set(text)

        if self._complete_vocabulary:
            for charset in CharTokenizer._charset:
                vocabulary.update(charset)

        vocabulary = sorted(vocabulary)
        return ({char: i for i, char in enumerate(vocabulary)},
                {i: char for i, char in enumerate(vocabulary)})
    
    def encode(self, text: str) -> list[int]:
        return [self._stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self._itos[id] for id in token_ids])
    
    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self._stoi)