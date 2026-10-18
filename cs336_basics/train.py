import argparse
import os
import time

import numpy as np
import torch
import wandb

from .transformer_lm import TransformerLM
from .adamw import AdamW
from .cross_entropy import cross_entropy
from .lr_schedule import get_lr_cosine_schedule
from .gradient_clipping import gradient_clipping
from .data_loading import get_batch
from .checkpointing import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing train, val")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")

    # model hyperparams
    parser.add_argument("--d_model", type=int, default=512, help="embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="feed forward dimension")
    parser.add_argument("--vocab_size", type=int, default=10000, help="vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="batch size per device")
    parser.add_argument("--max_lr", type=float, default=3e-3, help="max learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="minimum learning rate")
    parser.add_argument("--max_iters", type=int, default=4000, help="total number of training iterations")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="number of warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay for optimizer")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping threshold")

    # AdamW hyperparameters
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon")

    # device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use (cuda, mps, cpu)")
    parser.add_argument("--autocast_dtype", type=str, default="float32", choices=["float32", "bfloat16"],
                        help="bfloat16 runs matmuls in bf16 (weights, optimizer and loss stay float32)")
    parser.add_argument("--seed", type=int, default=0, help="seed for init and batch sampling")

    # logging
    parser.add_argument("--eval_interval", type=int, default=500, help="how often to evaluate on validation set")
    parser.add_argument("--eval_iters", type=int, default=200, help="number of batches per evaluation")
    parser.add_argument("--save_interval", type=int, default=1000, help="how often to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="how often to log metrics to console")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="tinystories-transformer", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging explicitly")

    return parser.parse_args()

@torch.no_grad()
def estimate_val_loss(model, val_data, batch_size, context_length, device, eval_iters, autocast):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        with autocast():
            logits = model(inputs)
        loss = cross_entropy(logits.float(), targets)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"using device: {device}")
    if device.type == "mps" and args.autocast_dtype == "bfloat16":
        raise ValueError("bfloat16 autocast with torch.compile gives nan gradients on mps (torch 2.11); use float32")
    torch.manual_seed(args.seed)

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.autocast_dtype == "bfloat16")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    tokens_per_step = args.batch_size * args.context_length

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
    model = torch.compile(model)

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay
    )

    iter_num = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"resuming from {args.resume}...")
        iter_num = load_checkpoint(args.resume, model, optimizer)

    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    train_start_time = time.time()
    t_iter_start = time.time()
    inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

    try:
        while iter_num < args.max_iters:
            lr = get_lr_cosine_schedule(iter_num, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with autocast():
                logits = model(inputs)
            loss = cross_entropy(logits.float(), targets)
            loss.backward()

            grad_norm = gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)

            is_last_iter = iter_num == args.max_iters - 1

            if iter_num % args.log_interval == 0:
                t_now = time.time()
                dt = t_now - t_iter_start
                t_iter_start = t_now

                loss_value = loss.item()
                ms_per_step = (dt * 1000) / args.log_interval

                print(f"iter {iter_num}: loss {loss_value:.4f}, time {ms_per_step:.2f}ms/step, lr {lr:.6f}")

                if use_wandb:
                    wandb.log({
                        "train/loss": loss_value,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "perf/ms_per_step": ms_per_step,
                        "wall_time": t_now - train_start_time,
                        "tokens": (iter_num + 1) * tokens_per_step
                    }, step=iter_num)

            if iter_num > 0 and (iter_num % args.eval_interval == 0 or is_last_iter):
                print(f"evaluating at iter {iter_num}...")
                val_loss = estimate_val_loss(
                    model, val_data, args.batch_size, args.context_length, device, args.eval_iters, autocast
                )
                print(f"step {iter_num}: val loss {val_loss:.4f}")

                if use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "wall_time": time.time() - train_start_time,
                        "tokens": (iter_num + 1) * tokens_per_step
                    }, step=iter_num)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(args.out_dir, 'ckpt_best.pt')
                    print(f"saving best checkpoint to {checkpoint_path}")
                    # store iterations completed, so resume starts at the next one
                    save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
                t_iter_start = time.time()

            if iter_num > 0 and (iter_num % args.save_interval == 0 or is_last_iter):
                checkpoint_path = os.path.join(args.out_dir, f'ckpt_{iter_num}.pt')
                print(f"saving regular checkpoint to {checkpoint_path}")
                save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)

            iter_num += 1
    except KeyboardInterrupt:
        print('training interrupted')
    finally:
        if use_wandb:
            wandb.finish()

    print("training completed")

if __name__ == "__main__":
    main()
