from pathlib import Path
from dataclasses import dataclass
import string

_CHARSETS = [" ", string.ascii_letters, string.digits, string.punctuation]

@dataclass
class TrainConfig:
    dataset_path = "data/tinyshakespeare.txt"


def load_text(path_to_textfile):
    with open(path_to_textfile, "r") as f:
        return f.read()


def build_alphabet_to_id(text):
    chars_in_text = set(text)

    alphabet = set()
    for charset in _CHARSETS:
        alphabet.update(charset)
    alphabet.update(chars_in_text)

    return {char: i for i, char in enumerate(sorted(alphabet))}


def main():
    config = TrainConfig()
    text = load_text(config.dataset_path)
    alphabet_to_id = build_alphabet_to_id(text)



if __name__ == "__main__":
    main()