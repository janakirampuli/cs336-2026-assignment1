import json
from collections.abc import Iterable, Iterator
from functools import lru_cache

import regex as re

# GPT-2 pre-tokenization regex: splits text into pre-tokens like "Hello", " world", "!", "\n\n"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

byte_tokens = [bytes([i]) for i in range(256)]


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte 0..255 to a printable character (the GPT-2 vocab/merges file format).

    Printable bytes keep their own character, e.g. 33 -> '!'. The other 68 (whitespace,
    control and non-ASCII bytes) are shifted to chr(256 + n), e.g. 32 (space) -> 'Ġ'.
    Every token then has a lossless text form with no spaces in it, so a merges line
    "Ġ t" always splits into exactly two tokens.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_to_char = {b: chr(b) for b in printable}
    n = 0
    for b in range(256):
        if b not in byte_to_char:
            byte_to_char[b] = chr(256 + n)
            n += 1
    return byte_to_char


@lru_cache
def unicode_to_bytes() -> dict[str, int]:
    # inverse of bytes_to_unicode, e.g. 'Ġ' -> 32
    return {char: b for b, char in bytes_to_unicode().items()}


def token_to_str(token: bytes) -> str:
    # e.g. b' the' -> 'Ġthe', b'\xe2' -> 'â'
    byte_to_char = bytes_to_unicode()
    return "".join(byte_to_char[b] for b in token)


def str_to_token(token_str: str) -> bytes:
    # inverse of token_to_str, e.g. 'Ġthe' -> b' the'
    char_to_byte = unicode_to_bytes()
    return bytes(char_to_byte[char] for char in token_str)

def merge_word(
        word: tuple[bytes, ...],
        pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    # e.g. word=(b' ', b't', b'h', b'e'), pair=(b'h', b'e') -> (b' ', b't', b'he')
    new_word = []
    left, right = pair
    merged_token = left + right
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == left and word[i+1] == right:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # vocab: token id -> token bytes, e.g. {0: b'!', ..., 256: b' t', ...}
        self.vocab: dict[int, bytes] = dict(vocab)
        # merges: (left, right) pairs in the order they were learned
        self.merges: list[tuple[bytes, bytes]] = merges
        # special_tokens: e.g. ["<|endoftext|>"], or [] when none are given
        self.special_tokens: list[str] = special_tokens if special_tokens else []
        # special_tokens_set: for fast membership checks, e.g. {"<|endoftext|>"}
        self.special_tokens_set: set[str] = set(self.special_tokens)

        # token_to_id: token bytes -> token id (vocab inverted), e.g. {b' t': 256, ...}
        self.token_to_id: dict[bytes, int] = {token: token_id for token_id, token in vocab.items()}
        # merge_ranks: pair -> position in merges; lower rank = learned earlier = applied first
        #   e.g. {(b' ', b't'): 0, (b'h', b'e'): 1, ...}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}

        # Special tokens missing from the vocab get new ids after the current largest one
        if self.special_tokens:
            next_id = (max(self.vocab.keys()) if self.vocab else -1) + 1
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.token_to_id:
                    self.vocab[next_id] = token_bytes
                    self.token_to_id[token_bytes] = next_id
                    next_id += 1

        self.pretoken_pattern = re.compile(PAT)

        self.special_split_pattern = None
        if self.special_tokens:
            longest_first = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(token) for token in longest_first)
            self.special_split_pattern = re.compile(f"({pattern_str})")

    def bpe_merge(self, tokens: tuple[bytes, ...]) -> tuple[bytes, ...]:
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            pair_ranks = {pair: self.merge_ranks[pair] for pair in pairs if pair in self.merge_ranks}

            if not pair_ranks:
                break

            best_pair = min(pair_ranks, key=pair_ranks.get)
            tokens = merge_word(tokens, best_pair)

        return tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load a tokenizer saved in the GPT-2 format (see bytes_to_unicode).

        vocab file: JSON of token string -> id, e.g. {"Ġthe": 262, ...}
        merges file: one merge per line, e.g. "Ġ t"
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            token_str_to_id = json.load(f)
        vocab = {token_id: str_to_token(token_str) for token_str, token_id in token_str_to_id.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                # GPT-2's own merges file starts with a "#version" header
                if not line or line.startswith("#version"):
                    continue

                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"{merges_filepath}:{line_num}: expected 2 tokens, got {line!r}")
                merges.append((str_to_token(parts[0]), str_to_token(parts[1])))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into a sequence of token ids.

        e.g. "Hello world" -> [15496, 995] (ids depend on the vocab)
        """
        ids = []

        # segments: text split around special tokens
        #   e.g. ["Hello, how ", "<|endoftext|>", " are you?", ...]
        if self.special_split_pattern:
            segments = self.special_split_pattern.split(text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            # special tokens map straight to their own id, never pre-tokenized or merged
            # (always in the vocab: __init__ adds any that were missing)
            if segment in self.special_tokens_set:
                ids.append(self.token_to_id[segment.encode("utf-8")])
                continue

            # pretokens: e.g. "Hello, how " -> ["Hello", ",", " how", " "]
            pretokens = self.pretoken_pattern.findall(segment)
            for pretoken in pretokens:
                # e.g. " the" -> (b' ', b't', b'h', b'e') -> merged -> (b' the',)
                tokens = self.bpe_merge(tuple(byte_tokens[b] for b in pretoken.encode("utf-8")))
                for token in tokens:
                    ids.append(self.token_to_id[token])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token ids for an iterable of strings (e.g. an open file handle).

        For files too large to load into memory at once: only one chunk (e.g. one line) is
        encoded at a time.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token ids back into text.

        e.g. [15496, 995] -> "Hello world"
        Byte sequences that aren't valid UTF-8 become U+FFFD instead of raising.
        """
        # text_bytes: all tokens' bytes joined, e.g. b'Hello' + b' world'. join() builds the result
        # once; `+=` on bytes copies the whole buffer every time, which is quadratic in length.
        text_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return text_bytes.decode("utf-8", errors="replace")
