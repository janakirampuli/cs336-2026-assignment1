import argparse

import torch

from .transformer_lm import TransformerLM
from .tokenizer import Tokenizer
from .checkpointing import load_checkpoint
from .softmax import softmax

def top_p_sampling(
        logits: torch.Tensor,
        top_p: float = 0.9,
        temperature: float = 1.0
) -> torch.Tensor:
    # temperature 0 means greedy: always take the most likely token
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    probs = softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cum_probs > top_p

        # shift right by one so the token that crosses top_p is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_id: int | None = None
) -> torch.Tensor:
    model.eval()
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        context_tokens = tokens[:, -context_length:]
        logits = model(context_tokens)

        logits = logits[:, -1, :]

        next_token = top_p_sampling(logits, top_p, temperature)

        tokens = torch.cat((tokens, next_token), dim=-1)

        if eos_id is not None and (next_token == eos_id).all():
            break

    return tokens

def main():
    parser = argparse.ArgumentParser()

    # model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="path to model checkpoint")

    # tokenizer
    parser.add_argument("--vocab_path", type=str, required=True, help="path to tokenizer vocab file")
    parser.add_argument("--merges_path", type=str, required=True, help="path to tokenizer merges file")

    # generation config
    parser.add_argument("--prompt", type=str, default="hi this is janaki, nice to meet you.", help="input text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="sampling temperature (lower is more deterministic, 0 is greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus sampling probability threshold")

    # model architecture, must match training (defaults match train.py)
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="feed forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # device
    parser.add_argument("--device", type=str, default="mps" if torch.mps.is_available() else "cpu",
                        help="device to use (cuda, mps, cpu)")

    args = parser.parse_args()
    device = torch.device(args.device)

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "theta": args.rope_theta,
        "device": device
    }

    print(f"TransformerLM config: {model_config}")
    model = TransformerLM(**model_config)
    model.to(device)

    print(f"loading from checkpoint {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model)

    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    print("generating...")

    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]

    generated_ids = generate(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id
    )

    output_ids = generated_ids.squeeze(0).tolist()
    decoded_text = tokenizer.decode(output_ids)

    print(decoded_text)

if __name__ == "__main__":
    main()
