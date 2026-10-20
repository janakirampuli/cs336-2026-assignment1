# Assignment 1 log

All training ran on Modal with bf16 autocast and `torch.compile`. The baseline model has d_model 512, 4 layers, 16 heads, d_ff 1344, RoPE theta 10000 and context length 256. On an H100 it runs at 28.5 ms per step at batch 128.

## TinyStories tokenizer and data

The 10K tokenizer compresses TinyStories to 4.12 bytes per token and encodes at about 0.93 MB/s in one process. Its longest tokens are words like " responsibility" and " disappointment".

| File | Tokens | Size |
|---|---|---|
| TinyStories train | 541,229,347 | 1.08 GB |
| TinyStories valid | 5,465,883 | 11 MB |

## Learning rate sweep (TinyStories)

Batch 128, 10,000 steps (327.68M tokens), 500 warmup steps, cosine decay to zero, AdamW beta2 0.95.

| Max lr | Val at step 5000 | Final val |
|---|---|---|
| 1e-4 | 1.900 | 1.807 |
| 3e-4 | 1.616 | 1.528 |
| 1e-3 | 1.492 | 1.386 |
| 3e-3 | 1.475 | 1.335 |
| 5e-3 | 1.495 | 1.325 |
| 7e-3 | 1.518 | 1.321 |
| 1e-2 | 1.548 | 1.321 |
| 1.5e-2 | 1.599 | 1.327 |
| 3e-2 | diverged | diverged at step 1210 |

Every value from 5e-3 to 1.5e-2 finished within 0.007 of the others, and each eval averages 50 batches, so differences under about 0.01 are noise.

The 3e-2 run had the lowest loss of any run at step 200. Once warmup pushed its learning rate past about 1.8e-2, the gradient norm grew from 0.25 to about 10 over 600 steps, jumped to 1.8e12 at step 1150, and the loss became `nan` at step 1210. Gradient clipping did not prevent it.

At step 5000 the lower learning rates were ahead, and the higher ones gained the most during the cosine decay at the end.

## Generated text (TinyStories)

The lr 1e-2 model (val 1.32), at temperature 0.8 and top-p 0.9:

> Once upon a time, there was a pretty girl named Lily. She had a pretty dress with lots of colors. She liked to play outside with her friends. One day, Lily's mom asked her to stay close to the big tree in the yard. Lily said, "Okay, Mom!" But she forgot to stay close to the tree. As the sun went down, Lily's pretty dress got dirty. She was very sad. Her mom said, "Don't worry, Lily. We will wash your dress." Lily was happy to hear that. She washed her dress and her mom gave her a big hug.

## Batch size (TinyStories)

Fixed budget of 327.68M tokens, so smaller batches took more steps. The first round scaled lr with the square root of the batch size from lr 1e-2 at batch 128. Validation always used 50 batches of 128 sequences.

| Batch | Lr | Final val | Time for the full budget |
|---|---|---|---|
| 1 | 9e-4 | 2.171 (on a capped 25.6M tokens) | about 4 hours, estimated |
| 8 | 2.5e-3 | 1.489 | about 55 min on an L4 |
| 32 | 5e-3 | 1.366 | 7.4 min |
| 32 | 1e-2 | 1.387 | 7.4 min |
| 64 | 7e-3 | 1.338 | 5.8 min |
| 128 | 1e-2 | 1.320 | 5.1 min |
| 256 | 1.4e-2 | 1.313 | 5.2 min |
| 512 | 2e-2 | 1.881 | 5.1 min |
| 512 | 1e-2 | 1.325 | 5.1 min |
| 512 | 7e-3 | 1.334 | 5.1 min |
| 1024 | 2.8e-2 | out of memory | n/a |

With lr 2e-2, batch 512 rose from 2.75 to 3.19 during warmup and never recovered, so the square root rule broke there. Batch 1024 ran out of memory on the 80 GB H100, on a 9.77 GiB allocation, which is the fp32 logits copy from `logits.float()` (1024 × 256 × 10,000 × 4 bytes).

## OpenWebText tokenizer

The 32K tokenizer trained on the 11.9 GB OpenWebText train file in 16.6 minutes on a 48 core CPU container, with a peak of 14 GB for the largest process.

| Tokenizer | Text | Bytes per token |
|---|---|---|
| TinyStories 10K | TinyStories | 4.11 |
| TinyStories 10K | OpenWebText | 3.21 |
| OpenWebText 32K | TinyStories | 4.00 |
| OpenWebText 32K | OpenWebText | 4.43 |

The longest OpenWebText tokens are runs of `.`, `*` and `-`, and the string `ÃÂÃÂ...` from text that was UTF-8 encoded twice.

The encoded train file has 2,727,120,452 tokens (5.45 GB), and the valid file has 66,401,098 tokens.

## OpenWebText run

Same setup as TinyStories (batch 128, 10,000 steps, lr 1e-2) with the 32K vocab, which makes the model 45.2M parameters.

| Dataset | Val loss per token | Bytes per token | Bits per byte | Perplexity |
|---|---|---|---|---|
| TinyStories | 1.313 | 4.12 | 0.46 | 3.7 |
| OpenWebText | 3.887 | 4.37 | 1.28 | 48.8 |

Bits per byte is the loss divided by the bytes per token and by ln 2.

Sample at temperature 0.8 and top-p 0.9:

> Once upon a time it was said that the list of these dates was finally complete, and had a time to complete. This year, the annual list is included in the list.
>
> “This is a really great time,” said Marcus Branson, an agency that recently became the first American city to enter the “city” country.
>
> “If you were really interested in the matter, you would have already given us our name, and if you were interested in that we would have given you the name of the year,” Branson said. “We are not a whole lot of the names. I’m not a part of that.” [...]

## Leaderboard run

The leaderboard limit is 45 minutes on a B200. We used one hour on an H100, with fused attention, where the baseline model runs at 22.7 ms per step.

### Model size

| Model (d_model × layers) | Params, total / non embedding | ms per step | Tokens in 42 min | Tokens per param, total / non embedding |
|---|---|---|---|---|
| 512 × 4 | 45M / 12.6M | 22.7 | 3.64B | 80 / 289 |
| 768 × 8 | 106M / 57M | 50.0 | 1.65B | 16 / 29 |
| 768 × 12 | 134M / 85M | 68.6 | 1.20B | 9 / 14 |
| 1024 × 8 | 167M / 101M | 71.3 | 1.16B | 7 / 11 |

We picked 768 × 8 with 12 heads and d_ff 2048, the closest to 20 tokens per parameter.

### Learning rate

10 minute runs of 11,500 steps each with the 768 × 8 model.

| Max lr | Val at step 1150 | Val at step 5750 | Final val |
|---|---|---|---|
| 3e-3 | 4.61 | 3.96 | 3.674 |
| 6e-3 | 4.89 | 4.06 | 3.678 (stopped at step 10,982 by the time limit) |
| 1e-2 | 5.76 | 4.28 | 3.780 |

### One hour run

lr 3e-3, 64,000 steps (2.1B tokens), 2,500 warmup steps, hard stop at 59 minutes. It finished in 55.7 minutes. The best eval was 3.384 at step 62,500.

| Run | Params | Tokens | Time | Final val | Bits per byte |
|---|---|---|---|---|---|
| Baseline model, 10,000 steps | 45M | 328M | about 4 min | 3.885 | 1.28 |
| Medium model, 10 minute trial | 106M | 377M | about 10 min | 3.674 | 1.21 |
| Medium model, one hour run | 106M | 2.1B | 55.7 min | 3.389 | 1.12 |

| Step | 5,000 | 15,000 | 25,000 | 40,000 | 50,000 | 60,000 | 63,999 |
|---|---|---|---|---|---|---|---|
| Val loss | 4.124 | 3.927 | 3.825 | 3.670 | 3.532 | 3.405 | 3.389 |

Sample at temperature 0.8 and top-p 0.9:

> Once upon a time, the story of the year is readable by the reader, and the audience is challenged, not just with the stories of the year itself, but with the history, both of the world and of the writing itself.
>
> The vision, as expressed by those who saw the book, was a vision for the future. There were many, many characters, a vision of the future, and it was a vision that was the foretaste of the future.
>
> It was a vision that had been in the memory of both J.R.R. Tolkien and the author of Tolkien’s Shadows, a vision that had been in the memory of both of the books. [...]
