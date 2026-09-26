import argparse
import os
from typing import List, Dict, Tuple, BinaryIO, Optional
import multiprocessing
import time
from collections import Counter, defaultdict
from pathlib import Path
import regex as re
import cProfile
import pstats
import json
import heapq


# GPT-2 pre-tokenization regex: splits text into pre-tokens like "Hello", " world", "!", "\n\n"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class ReversedPair:
    """Wraps a pair so it sorts in reverse: ReversedPair(a) < ReversedPair(b) iff a > b.

    heapq is a min-heap, and on count ties we want the lexicographically greatest pair
    first. Counts can be negated but bytes can't, so the pair is wrapped instead.
    e.g. heap entries (-7, ReversedPair((b't', b'h'))) and (-7, ReversedPair((b' ', b't'))):
    (b't', b'h') > (b' ', b't'), so (b't', b'h') pops first.
    """
    __slots__ = ("pair",)  # no per-instance __dict__: less memory for many heap entries

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
        input_path: str,
        start: int,
        end: int,
        special_tokens: Optional[List[str]]
) -> Counter:
    # word_freqs: pre-token (as a tuple of single-byte tokens) -> count in this chunk
    #   e.g. {(b' ', b't', b'h', b'e'): 5821, (b'<|endoftext|>',): 312, ...}
    word_freqs = Counter()

    pretoken_pattern = re.compile(PAT)
    # empty set when there are no special tokens, e.g. {"<|endoftext|>"}
    special_tokens_set = set(special_tokens or [])
    special_split_pattern = None

    if special_tokens:
        # e.g. "(<\|endoftext\|>)"; the capture group keeps the special tokens in split() output
        pattern_str = "|".join(re.escape(t) for t in special_tokens)
        special_split_pattern = re.compile(f"({pattern_str})")

    with open(input_path, 'rb') as f:
        f.seek(start)
        size_to_read = end - start

        if size_to_read <= 0:
            return word_freqs

        chunk_bytes = f.read(size_to_read)
        text = chunk_bytes.decode('utf-8', errors="ignore")

        # segments: text split around special tokens
        #   e.g. ["Once upon a time...", "<|endoftext|>", "One day...", ...]
        if special_split_pattern:
            segments = special_split_pattern.split(text)
        else:
            segments = [text]

        for segment in segments:
            if not segment:
                continue

            if segment in special_tokens_set:
                # (don't split into bytes)
                # e.g., (b'<|endoftext|>',)
                word_freqs[(segment.encode('utf-8'),)] += 1
                continue

            # pretokens: e.g. ["Once", " upon", " a", " time", "."]
            pretokens = pretoken_pattern.findall(segment)
            word_freqs.update(
                tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                for pretoken in pretokens
            )
    return word_freqs

def merge_word(
        word: Tuple[bytes, ...],
        pair: Tuple[bytes, bytes]
) -> Tuple[bytes, ...]:
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


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    num_procs = os.cpu_count()

    # boundaries: byte offsets of chunk edges, each chunk starting at a special token
    #   e.g. [0, 2181423, 4362931, ..., file_size]
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_procs, b"<|endoftext|>")

    # chunk_args: one (input_path, start, end, special_tokens) tuple per worker
    chunk_args = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))

    # word_freqs: word (tuple of single-byte tokens) -> count over the whole corpus
    #   e.g. {(b' ', b't', b'h', b'e'): 58210, ...}
    word_freqs = Counter()

    with multiprocessing.Pool(processes=num_procs) as pool:
        per_chunk_word_freqs = pool.starmap(process_chunk, chunk_args)

        for chunk_word_freqs in per_chunk_word_freqs:
            word_freqs.update(chunk_word_freqs)

    # word_tokens[word_id]: the word's current segmentation, rewritten in place by merges
    # e.g. word_tokens[0] = (b' ', b't', b'h', b'e') -> (b' ', b't', b'he') -> (b' the',)
    # word_counts[word_id]: how often the word occurs
    word_tokens = []
    word_counts = []

    for word, freq in word_freqs.items():
        word_tokens.append(word)
        word_counts.append(freq)

    del word_freqs

    # pair_counts: adjacent token pair -> total count across all words, weighted by word count
    #   e.g. {(b' ', b't'): 90210, (b't', b'h'): 71553, ...}
    pair_counts = Counter()
    # pair_to_word_ids: pair -> ids of words that contain it (or used to; see lazy removal below)
    #   e.g. {(b't', b'h'): {0, 1, 57}, ...}
    pair_to_word_ids = defaultdict(set)

    for word_id, tokens in enumerate(word_tokens):
        count = word_counts[word_id]
        for pos in range(len(tokens) - 1):
            pair = (tokens[pos], tokens[pos+1])
            pair_counts[pair] += count
            pair_to_word_ids[pair].add(word_id)

    pair_heap = [(-count, ReversedPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)  # O(n), cheaper than n pushes

    merges = []

    # vocab_tokens: token id -> token bytes, as a list where the index is the id
    #   e.g. [b'\x00', ..., b'\xff', b'<|endoftext|>', b' t', b'he', ...]
    vocab_tokens = [bytes([i]) for i in range(256)]
    for token in special_tokens:
        vocab_tokens.append(token.encode("utf-8"))

    num_merges = vocab_size - len(vocab_tokens)

    for _ in range(num_merges):
        best_pair = None
        while pair_heap:
            neg_count, wrapped = heapq.heappop(pair_heap)
            pair = wrapped.pair
            if pair_counts.get(pair) == -neg_count:
                best_pair = pair
                break
        if best_pair is None:
            break

        merges.append(best_pair)
        vocab_tokens.append(best_pair[0] + best_pair[1])

        # changed_pairs: every pair whose count went up or down during this merge,
        #   e.g. {(b' ', b't'), (b'h', b'e'), (b' ', b'th'), (b'th', b'e')}
        changed_pairs = set()
        # candidate_ids: words listed under best_pair, e.g. {0, 1, 57}
        candidate_ids = pair_to_word_ids.pop(best_pair, set())

        for word_id in candidate_ids:
            old_tokens = word_tokens[word_id]
            new_tokens = merge_word(old_tokens, best_pair)
            # lazy removal: the index is never cleaned up when a word loses a pair, so a candidate
            # may no longer contain best_pair
            if new_tokens == old_tokens:
                continue

            count = word_counts[word_id]
            # remove this word's old pairs
            for pos in range(len(old_tokens) - 1):
                pair = (old_tokens[pos], old_tokens[pos+1])
                pair_counts[pair] -= count
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                changed_pairs.add(pair)

            # add its new ones, registering the word under each new pair
            for pos in range(len(new_tokens) - 1):
                pair = (new_tokens[pos], new_tokens[pos+1])
                pair_counts[pair] += count
                pair_to_word_ids[pair].add(word_id)
                changed_pairs.add(pair)

            word_tokens[word_id] = new_tokens

        for pair in changed_pairs:
            count = pair_counts.get(pair)
            if count:
                heapq.heappush(pair_heap, (-count, ReversedPair(pair)))

    # vocab: token id -> token bytes, e.g. {0: b'\x00', ..., 256: b'<|endoftext|>', 257: b' t', ...}
    vocab = {idx: token for idx, token in enumerate(vocab_tokens)}
    return vocab, merges

def save_to_disk(
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        vocab_path="vocab.json",
        merges_path="merges.txt"
):
    # vocab_strs: token id -> token as text (lossy: invalid UTF-8 bytes become U+FFFD)
    vocab_strs = {
        k: v.decode('utf-8', errors='replace') for k, v in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_strs, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.decode('utf-8', errors='replace')} {right.decode('utf-8', errors='replace')}\n")


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--profile", action="store_true",
                        help="run under cProfile and save profiles/train_bpe.prof")
    parser.add_argument("--top", type=int, default=20, help="rows to print per sort order")
    args = parser.parse_args()

    # input_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    input_path =  "./data/TinyStoriesV2-GPT4-train.txt"

    profiler = cProfile.Profile() if args.profile else None

    start_time = time.perf_counter()
    if profiler:
        profiler.enable()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )

    if profiler:
        profiler.disable()
    time_taken = time.perf_counter() - start_time

    print(f"vocab size: {len(vocab)}")
    print(f"merges size: {len(merges)}")

    print(f"total time taken for tokenisation: {time_taken:.4f} seconds ({time_taken/60:.4f} mins)")

    if profiler:
        out_dir = Path("profiles")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "train_bpe.prof"
        profiler.dump_stats(out_file)
        # tottime = time in the function body itself; cumtime = including everything it calls
        for sortby in ("tottime", "cumtime"):
            print(f"\n--- top {args.top} by {sortby} ---")
            pstats.Stats(str(out_file)).sort_stats(sortby).print_stats(args.top)
        print(f"saved {out_file}  ->  re-analyse without re-running: uv run python -m pstats {out_file}")

    save_to_disk(vocab, merges)

if __name__ == "__main__":
    main()
