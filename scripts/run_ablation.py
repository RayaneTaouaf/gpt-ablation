"""
Ablation: pre-norm vs post-norm.

Protocol: two STRICTLY identical models (same seed, same config, same parameter
count, same learning-rate schedule with warmup), differing only in LayerNorm
placement. We measure how many iterations each needs to reach a given
validation-loss threshold.
"""
import argparse, json, sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data import load_data, get_batch
from src.model import GPT
from src.train import train


def steps_to_threshold(history, threshold):
    """First iteration where val loss <= threshold. None if never reached."""
    for rec in history:
        if rec["val"] <= threshold:
            return rec["iter"]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}\n")

    tok, train_data, val_data = load_data("data/tinyshakespeare.txt", device=device)

    results = {}
    for style in ("pre", "post"):
        print(f"=== norm_style = {style} ===")
        torch.manual_seed(args.seed)   # SAME seed: same init, same batches
        model = GPT(vocab_size=tok.vocab_size, n_embd=args.n_embd,
                    n_head=args.n_head, n_layer=args.n_layer,
                    block_size=args.block_size, norm_style=style)
        print(f"  params: {model.num_params():,}")
        hist = train(model, train_data, val_data, get_batch,
                     max_iters=args.max_iters, batch_size=args.batch_size,
                     block_size=args.block_size, lr=args.lr,
                     warmup_iters=args.warmup, eval_interval=args.eval_interval,
                     device=device, run_name=f"{style}norm", use_wandb=args.wandb)
        results[style] = hist
        print()

    # --- Analysis ---
    final_pre = results["pre"][-1]["val"]
    final_post = results["post"][-1]["val"]
    # Threshold: the final loss of the SLOWER of the two. By construction both
    # reach it, so the comparison is fair: "how many iterations to reach the
    # same quality level?".
    threshold = max(final_pre, final_post)

    s_pre = steps_to_threshold(results["pre"], threshold)
    s_post = steps_to_threshold(results["post"], threshold)

    print("=" * 62)
    print(f"final val loss  pre-norm  : {final_pre:.4f}")
    print(f"final val loss  post-norm : {final_post:.4f}")
    print(f"\nComparison threshold: val loss <= {threshold:.4f}")
    print(f"  pre-norm  reached the threshold in {s_pre} iterations")
    print(f"  post-norm reached the threshold in {s_post} iterations")
    if s_pre and s_post and s_pre > 0:
        speedup = s_post / s_pre
        print(f"\n>>> pre-norm converges {speedup:.2f}x faster than post-norm")
    print("=" * 62)

    summary = {"threshold": threshold, "final_val_pre": final_pre,
               "final_val_post": final_post, "steps_pre": s_pre,
               "steps_post": s_post,
               "speedup": (s_post / s_pre) if (s_pre and s_post) else None,
               "config": vars(args)}
    with open("results/ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary written to results/ablation_summary.json")


if __name__ == "__main__":
    main()
