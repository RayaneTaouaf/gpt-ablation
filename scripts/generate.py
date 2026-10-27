"""Train a model and generate text. Also measures the KV-cache speedup."""
import argparse, sys, pathlib, time, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.data import load_data, get_batch
from src.model import GPT
from src.train import train

p = argparse.ArgumentParser()
p.add_argument("--max-iters", type=int, default=2000)
p.add_argument("--n-embd", type=int, default=256)
p.add_argument("--n-head", type=int, default=8)
p.add_argument("--n-layer", type=int, default=6)
p.add_argument("--block-size", type=int, default=128)
p.add_argument("--tokens", type=int, default=400)
p.add_argument("--wandb", action="store_true")
a = p.parse_args()

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
torch.manual_seed(1337)

tok, tr, va = load_data("data/tinyshakespeare.txt", device=device)
model = GPT(vocab_size=tok.vocab_size, n_embd=a.n_embd, n_head=a.n_head,
            n_layer=a.n_layer, block_size=a.block_size, norm_style="pre")
print(f"params: {model.num_params():,}")
train(model, tr, va, get_batch, max_iters=a.max_iters, block_size=a.block_size,
      device=device, run_name="generate", use_wandb=a.wandb)

ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

# Speed comparison: with and without the KV cache
t0 = time.time(); out = model.generate(ctx, a.tokens, top_k=50, use_cache=True)
t_cache = time.time() - t0
t0 = time.time(); model.generate(ctx, a.tokens, top_k=50, use_cache=False)
t_nocache = time.time() - t0
print(f"\ngenerating {a.tokens} tokens:")
print(f"  with KV cache : {t_cache:.2f}s")
print(f"  no cache      : {t_nocache:.2f}s")
print(f"  speedup       : {t_nocache/t_cache:.2f}x\n")
print(tok.decode(out[0].tolist()))
