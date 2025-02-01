from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.

    A tokenizer is an object that takes a string and converts it into a
    sequence of tokens, which are integers that represent the string.
    """

    @abstractmethod
    def train(self, text: str) -> None:
        """
        Train the tokenizer on the given text.

        This method must be called before any of the other methods can be
        used. It is used to learn the vocabulary of the tokenizer.
        """
        ...

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Convert a string into a sequence of token ids.

        Args:
            text: String to be converted into a sequence of token ids.

        Returns:
            A list of token ids.
        """
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a sequence of token ids back into the original string.

        Args:
            token_ids: List of token ids to be converted back into a string.

        Returns:
            The original string.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the vocabulary size.
        """
        ...


