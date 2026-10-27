"""Training loop with warmup + cosine decay, and optional W&B logging."""

import json
import math
import time
from pathlib import Path

import torch


def get_lr(it, max_iters, lr, warmup_iters, min_lr_ratio=0.1):
    """
    Learning rate: linear warmup then cosine decay.

    Warmup is essential here: without it, post-norm often diverges in the first
    few steps. We apply it to BOTH variants so the ablation stays honest: we do
    not deliberately handicap post-norm.
    """
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, get_batch, batch_size,
                  block_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, get_batch, *,
          max_iters=3000, batch_size=32, block_size=256,
          lr=3e-4, warmup_iters=100, grad_clip=1.0,
          eval_interval=250, eval_iters=100,
          device="cpu", run_name="run", use_wandb=False,
          out_dir="results", verbose=True):

    model = model.to(device)

    # No weight decay on biases and LayerNorm: standard convention, these
    # parameters should not be pulled toward zero.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="gpt-norm-ablation", name=run_name,
                                   config={"norm_style": model.norm_style,
                                           "max_iters": max_iters, "lr": lr,
                                           "batch_size": batch_size,
                                           "block_size": block_size})
        except Exception as e:
            print(f"W&B unavailable ({e}), continuing without it.")

    history = []
    t0 = time.time()

    for it in range(max_iters + 1):
        cur_lr = get_lr(it, max_iters, lr, warmup_iters)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, train_data, val_data, get_batch,
                                   batch_size, block_size, eval_iters, device)
            rec = {"iter": it, "lr": cur_lr, "elapsed": time.time() - t0, **losses}
            history.append(rec)
            if verbose:
                print(f"  iter {it:5d} | train {losses['train']:.4f} | "
                      f"val {losses['val']:.4f} | lr {cur_lr:.2e} | "
                      f"{rec['elapsed']:.0f}s")
            if wandb_run:
                wandb_run.log({"iter": it, "train_loss": losses["train"],
                               "val_loss": losses["val"], "lr": cur_lr})

        if it == max_iters:
            break

        xb, yb = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clipping: bounds the global gradient norm. Without it, post-norm can
        # blow up on a bad batch.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if wandb_run:
        wandb_run.finish()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{run_name}.json", "w") as f:
        json.dump({"run_name": run_name, "norm_style": model.norm_style,
                   "n_params": model.num_params(), "history": history}, f, indent=2)

    return history
