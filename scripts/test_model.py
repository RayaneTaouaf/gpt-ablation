import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.model import GPT

torch.manual_seed(0)
V, B, T = 65, 2, 16
m = GPT(vocab_size=V, n_embd=64, n_head=4, n_layer=2, block_size=32)
m.eval()

# 1. Shapes + initial loss ~ ln(V)
x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
logits, loss = m(x, y)
import math
print(f"1. logits {tuple(logits.shape)} | loss {loss.item():.4f} | ln(V)={math.log(V):.4f}")

# 2. CAUSALITY: changing token t must NOT change outputs at positions < t
x1 = torch.randint(0, V, (1, T))
x2 = x1.clone(); x2[0, -1] = (x2[0, -1] + 1) % V   # change the LAST token
with torch.no_grad():
    o1, _ = m(x1); o2, _ = m(x2)
diff_before = (o1[:, :-1] - o2[:, :-1]).abs().max().item()
diff_last   = (o1[:, -1]  - o2[:, -1]).abs().max().item()
print(f"2. causality: diff positions<t = {diff_before:.2e} (should be ~0) | diff last = {diff_last:.4f} (should be >0)")

# 3. KV CACHE: generation with and without cache must give the SAME result
torch.manual_seed(42); a = m.generate(torch.zeros((1,1),dtype=torch.long), 20, use_cache=True)
torch.manual_seed(42); b = m.generate(torch.zeros((1,1),dtype=torch.long), 20, use_cache=False)
print(f"3. KV cache matches no-cache: {torch.equal(a,b)}")
if not torch.equal(a,b):
    print("   with cache:", a.tolist()); print("   no cache:", b.tolist())

# 4. pre vs post norm build correctly
for style in ("pre","post"):
    mm = GPT(vocab_size=V, n_embd=64, n_head=4, n_layer=2, block_size=32, norm_style=style)
    _, l = mm(x, y)
    print(f"4. norm_style={style:4s} params={mm.num_params():,} loss={l.item():.4f}")
