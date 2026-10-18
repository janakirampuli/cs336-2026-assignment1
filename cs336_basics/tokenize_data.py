import argparse
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import Tokenizer
from .train_bpe import find_chunk_boundaries

# set once per worker by init_worker, so the tokenizer is loaded once per process, not per chunk
worker_tokenizer: Tokenizer | None = None


def init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global worker_tokenizer
    worker_tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def encode_chunk(args: tuple[str, int, int]) -> np.ndarray:
    # chunk boundaries sit on special tokens, so no character or document is split across chunks
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="replace")
    return np.array(worker_tokenizer.encode(text), dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Encode a text file into a flat uint16 array of token ids")
    parser.add_argument("--input_path", type=str, required=True, help="text file to encode")
    parser.add_argument("--output_path", type=str, required=True, help="where to write the .bin file")
    parser.add_argument("--vocab_path", type=str, default="vocab.json", help="tokenizer vocab file")
    parser.add_argument("--merges_path", type=str, default="merges.txt", help="tokenizer merges file")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="special tokens; the first one is also used to split the file into chunks")
    parser.add_argument("--num_procs", type=int, default=os.cpu_count(), help="number of worker processes")
    parser.add_argument("--chunk_mb", type=int, default=16,
                        help="target chunk size; smaller chunks use less memory per worker")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)
    vocab_size = len(tokenizer.vocab)
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab size {vocab_size} does not fit in uint16")

    file_size = os.path.getsize(args.input_path)
    num_chunks = max(1, file_size // (args.chunk_mb * 2**20))
    with open(args.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, args.special_tokens[0].encode("utf-8"))
    chunks = [(args.input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    print(f"encoding {args.input_path} ({file_size / 1e9:.2f} GB) in {len(chunks)} chunks "
          f"on {args.num_procs} processes, vocab size {vocab_size}")

    # write to a temp file and rename at the end, so a crashed run never leaves a half-written .bin
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    start_time = time.perf_counter()
    num_tokens = 0
    with (
        multiprocessing.Pool(
            args.num_procs, initializer=init_worker,
            initargs=(args.vocab_path, args.merges_path, args.special_tokens)
        ) as pool,
        open(tmp_path, "wb") as out,
    ):
        # imap keeps chunk order, so tokens are written in the same order as the text
        for ids in tqdm(pool.imap(encode_chunk, chunks), total=len(chunks), unit="chunk"):
            ids.tofile(out)
            num_tokens += len(ids)
    tmp_path.rename(output_path)
    time_taken = time.perf_counter() - start_time

    print(f"wrote {num_tokens:,} tokens to {output_path} ({num_tokens * 2 / 1e9:.2f} GB)")
    print(f"{file_size / num_tokens:.2f} bytes/token | {time_taken:.1f}s | {file_size / time_taken / 1e6:.2f} MB/s")


if __name__ == "__main__":
    main()
