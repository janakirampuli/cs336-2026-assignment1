import json
from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 pre-tokenization regex: splits text into pre-tokens like "Hello", " world", "!", "\n\n"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

byte_tokens = [bytes([i]) for i in range(256)]

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
        with open(vocab_filepath, 'r') as f:
            vocab_strs = json.load(f)
        vocab = {int(token_id): token.encode("utf-8") for token_id, token in vocab_strs.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

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
