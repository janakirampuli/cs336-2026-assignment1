# Assignment 1 experiments

The notes below log the experiments we ran, from the tokenizer results to the leaderboard run, in the order we ran them. The commands we used are in `commands.txt`.

## Setup

All training ran on Modal GPUs with bf16 autocast and `torch.compile`, and runs log to Weights and Biases, in the `tinystories-transformer` and `owt-transformer` projects. The model is the handout's baseline unless a section says otherwise. It has d_model 512, 4 layers, 16 heads, d_ff 1344, RoPE theta 10000, and a context length of 256.

| GPU | Batch | ms per step |
|---|---|---|
| M5 Mac (fp32) | 32 | about 1,150 |
| L4 (bf16) | 32 | 66 |
| L4 (bf16) | 128 | 250 |
| H100 (bf16) | 128 | 28.5 |

The H100 is 8.8 times faster than the L4 at batch 128, and a full run of the handout's 327.68M tokens takes about 5 minutes on it.

## TinyStories tokenizer and data

We trained the 10K tokenizer on the TinyStories train file. Its longest tokens are words like " responsibility" and " disappointment", and it compresses TinyStories to 4.12 bytes per token. It encodes at about 0.93 MB/s in one process.

We stored the encoded data as flat uint16 arrays. Every token id is below 65,536, so uint16 is enough, and it takes half the space of int32.

| File | Tokens | Size |
|---|---|---|
| TinyStories train | 541,229,347 | 1.08 GB |
| TinyStories valid | 5,465,883 | 11 MB |

## Learning rate sweep on TinyStories

We trained at batch size 128 for 10,000 steps, which is the handout's full budget of 327.68M tokens. Each run used 500 warmup steps, a cosine schedule that decays to zero, and AdamW with beta2 0.95. We ran the coarse sweep first and then a narrow sweep around the best value.

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

The learning rate matters a lot across orders of magnitude and very little near the best value. Going from 1e-4 to 1e-3 lowered the final loss by 0.42, while every value from 5e-3 to 1.5e-2 finished within 0.007 of the others. Each eval averages 50 batches, so differences under about 0.01 are noise.

The best learning rate sits just below the point where training breaks. The 3e-2 run had the lowest loss of any run at step 200, while warmup still held its learning rate near 1e-2. Once warmup pushed the learning rate past about 1.8e-2, the loss started to rise, and the gradient norm grew from 0.25 to about 10 over the next 600 steps. The norm then jumped to 1.8e12 at step 1150, and the loss became `nan` at step 1210.

Gradient clipping did not prevent the divergence. AdamW divides each update by a running average of the gradient size, so each step moves the weights by roughly the learning rate however large the gradient is. Shrinking the gradient changes little once the learning rate itself is too large.

High learning rates looked worse in the middle of training and finished best. At step 5000, lower learning rates were ahead in every run, but the cosine schedule lowered the learning rate at the end, and the runs with large learning rates gained the most during that decay.

## Generated text from TinyStories

The lr 1e-2 model, with validation loss 1.32, wrote the story below at temperature 0.8 and top-p 0.9.

> Once upon a time, there was a pretty girl named Lily. She had a pretty dress with lots of colors. She liked to play outside with her friends. One day, Lily's mom asked her to stay close to the big tree in the yard. Lily said, "Okay, Mom!" But she forgot to stay close to the tree. As the sun went down, Lily's pretty dress got dirty. She was very sad. Her mom said, "Don't worry, Lily. We will wash your dress." Lily was happy to hear that. She washed her dress and her mom gave her a big hug.
## Batch size experiment

We kept the token budget fixed at 327.68M tokens, so smaller batches took more steps. For the first round we scaled the learning rate with the square root of the batch size, starting from lr 1e-2 at batch 128. We then reran the batch sizes where the scaled learning rate looked wrong. Validation always used 50 batches of 128 sequences, so every run measured validation loss on the same amount of data.

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

Loss per token improves with batch size up to about 128 and then stays flat through 512. Very small batches are much worse, because a batch of one 256 token sequence gives a very noisy gradient.

Bigger batches are faster in wall clock time up to the memory limit. Every batch size from 128 up finished the budget in about 5 minutes on an H100, while batch 32 needed 7.4 minutes because the GPU sat partly idle.

The square root rule broke at batch 512. With lr 2e-2, the validation loss rose from 2.75 to 3.19 during warmup and never recovered within 2,500 steps, but the same batch size with lr 1e-2 reached 1.325. The best learning rate stayed near 1e-2 from batch 128 to batch 512, so the stable range did not grow as fast as the rule assumes.

Raising the learning rate made batch 32 worse, from 1.366 to 1.387, so a badly tuned learning rate does not explain why small batches learn less per token here. One possible reason is weight decay, because AdamW shrinks the weights by `lr × weight_decay` on every step, and batch 32 takes four times as many steps. We did not test that idea.

Batch 1024 ran out of memory on the 80 GB H100. The failed allocation was 9.77 GiB, which is the fp32 copy of the logits that `logits.float()` makes, at 1024 × 256 × 10,000 values of 4 bytes each.

## OpenWebText tokenizer

We trained the 32K tokenizer on the 11.9 GB OpenWebText train file, on a CPU container with 48 cores. Training took 16.6 minutes, and the largest process peaked at 14 GB of memory during the merge phase.

| Tokenizer | Text | Bytes per token |
|---|---|---|
| TinyStories 10K | TinyStories | 4.11 |
| TinyStories 10K | OpenWebText | 3.21 |
| OpenWebText 32K | TinyStories | 4.00 |
| OpenWebText 32K | OpenWebText | 4.43 |

A tokenizer compresses text best when the text is like its training data. The TinyStories tokenizer has to split web words into many small pieces, so OpenWebText needs 22% more tokens with it. The OpenWebText tokenizer loses only a little on TinyStories, because simple words also appear often in web text.

The longest OpenWebText tokens are runs of `.`, `*`, and `-`, and the string `ÃÂÃÂ...`, which comes from text that was UTF-8 encoded twice somewhere on the web. BPE merges whatever repeats most often, so junk that repeats often still takes up vocabulary slots.

The encoded train file has 2,727,120,452 tokens (5.45 GB), and the valid file has 66,401,098 tokens.

## OpenWebText run

We trained on OpenWebText with the TinyStories setup, which is batch 128, 10,000 steps, and lr 1e-2, and only the data and the vocab size changed. The model has 45.2M parameters, because the embedding and the lm_head each grow from 5.1M to 16.4M parameters with a 32K vocab. The final validation loss was 3.887.

Loss per token cannot be compared across tokenizers, so we converted both losses to bits per byte. The conversion divides the loss by the bytes per token and then by ln 2.

| Dataset | Val loss per token | Bytes per token | Bits per byte | Perplexity |
|---|---|---|---|---|
| TinyStories | 1.313 | 4.12 | 0.46 | 3.7 |
| OpenWebText | 3.887 | 4.37 | 1.28 | 48.8 |

Per byte, OpenWebText is about 2.8 times harder for the same model and compute. OpenWebText covers many topics with rare words, and the run saw only 12% of it once, while the TinyStories run saw 60% of a much simpler dataset.

Sample from the prompt "Once upon a time", at temperature 0.8 and top-p 0.9:

> Once upon a time it was said that the list of these dates was finally complete, and had a time to complete. This year, the annual list is included in the list.
>
> “This is a really great time,” said Marcus Branson, an agency that recently became the first American city to enter the “city” country.
>
> “If you were really interested in the matter, you would have already given us our name, and if you were interested in that we would have given you the name of the year,” Branson said. “We are not a whole lot of the names. I’m not a part of that.” [...]

## Leaderboard run

The leaderboard asks for the lowest OpenWebText validation loss within 45 minutes on a B200. We used one hour on an H100, which is a slower GPU. These runs used fused attention, and the baseline model ran at 22.7 ms per step.

### Choosing the model size

At 22.7 ms per step, the baseline model would see about 3.6B tokens in 42 minutes, which is more than all of OpenWebText and about 80 tokens per parameter. A common rule from the Chinchilla paper says compute is used best at about 20 tokens per parameter, so we measured the speed of larger models.

| Model (d_model × layers) | Params, total / non embedding | ms per step | Tokens in 42 min | Tokens per param, total / non embedding |
|---|---|---|---|---|
| 512 × 4 | 45M / 12.6M | 22.7 | 3.64B | 80 / 289 |
| 768 × 8 | 106M / 57M | 50.0 | 1.65B | 16 / 29 |
| 768 × 12 | 134M / 85M | 68.6 | 1.20B | 9 / 14 |
| 1024 × 8 | 167M / 101M | 71.3 | 1.16B | 7 / 11 |

We picked 768 × 8 with 12 heads and d_ff 2048, because it is closest to 20 tokens per parameter and its token count fits inside OpenWebText. The 1024 × 8 model has 19% more non embedding parameters than the 768 × 12 model and runs at almost the same speed, because wider matrix multiplies keep the H100 busier than extra layers do.

### Learning rate for the medium model

We tuned the learning rate with three 10 minute runs of 11,500 steps each.

| Max lr | Val at step 1150 | Val at step 5750 | Final val |
|---|---|---|---|
| 3e-3 | 4.61 | 3.96 | 3.674 |
| 6e-3 | 4.89 | 4.06 | 3.678 (stopped at step 10,982 by the time limit) |
| 1e-2 | 5.76 | 4.28 | 3.780 |

The larger model needs a lower learning rate. The small model's best value of 1e-2 is clearly too high for the medium model, which trails at every eval, while 3e-3 and 6e-3 finish about tied. In 10 minutes the medium model reached 3.674, compared with 3.885 for the small model's full run.

### One hour run

The final run used the medium model with lr 3e-3, 64,000 steps (2.1B tokens, about 20 tokens per parameter), 2,500 warmup steps, and a hard stop at 59 minutes. It finished all 64,000 steps in 55.7 minutes, with a final validation loss of 3.389. The best eval was 3.384 at step 62,500, and the difference from the final value is within eval noise.

| Run | Params | Tokens | Time | Final val | Bits per byte |
|---|---|---|---|---|---|
| Baseline model, 10,000 steps | 45M | 328M | about 4 min | 3.885 | 1.28 |
| Medium model, 10 minute trial | 106M | 377M | about 10 min | 3.674 | 1.21 |
| Medium model, one hour run | 106M | 2.1B | 55.7 min | 3.389 | 1.12 |

| Step | 5,000 | 15,000 | 25,000 | 40,000 | 50,000 | 60,000 | 63,999 |
|---|---|---|---|---|---|---|---|
| Val loss | 4.124 | 3.927 | 3.825 | 3.670 | 3.532 | 3.405 | 3.389 |

The loss fell slowly through the middle of the run and faster near the end. From step 15,000 to step 40,000 it dropped by 0.26, and over the last 24,000 steps it dropped by another 0.28, because the cosine schedule lowered the learning rate in that stretch. The same pattern showed up in the TinyStories sweep.

Sample from the prompt "Once upon a time", at temperature 0.8 and top-p 0.9:

> Once upon a time, the story of the year is readable by the reader, and the audience is challenged, not just with the stories of the year itself, but with the history, both of the world and of the writing itself.
>
> The vision, as expressed by those who saw the book, was a vision for the future. There were many, many characters, a vision of the future, and it was a vision that was the foretaste of the future.
>
> It was a vision that had been in the memory of both J.R.R. Tolkien and the author of Tolkien’s Shadows, a vision that had been in the memory of both of the books. [...]
