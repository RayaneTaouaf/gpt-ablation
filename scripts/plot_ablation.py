"""Plot validation curves for both variants."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pre = json.load(open("results/prenorm.json"))["history"]
post = json.load(open("results/postnorm.json"))["history"]
summ = json.load(open("results/ablation_summary.json"))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([h["iter"] for h in pre],  [h["val"] for h in pre],  label="pre-norm",  lw=2)
ax.plot([h["iter"] for h in post], [h["val"] for h in post], label="post-norm", lw=2)
ax.axhline(summ["threshold"], ls="--", c="grey", lw=1,
           label=f"threshold {summ['threshold']:.3f}")
if summ["steps_pre"]:
    ax.axvline(summ["steps_pre"], ls=":", c="C0", lw=1)
if summ["steps_post"]:
    ax.axvline(summ["steps_post"], ls=":", c="C1", lw=1)
ax.set_xlabel("iteration"); ax.set_ylabel("validation loss")
title = "Pre-norm vs post-norm"
if summ.get("speedup"):
    title += f"  pre-norm {summ['speedup']:.2f}x faster"
ax.set_title(title)
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("results/ablation.png", dpi=150)
print("wrote results/ablation.png")
