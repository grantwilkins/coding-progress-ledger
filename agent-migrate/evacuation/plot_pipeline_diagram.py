"""Queue-Haul ordered-pipeline diagram (Fig. pipeline). Reproducible source for
queue-haul-pipeline.pdf; box labels track the formulation's Stage 1-4 names."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "ee364b-write-up"
TEAL = "#1a635a"
STAGES = [
    "Proportional\nfairness\nallocator",
    "Minimize\npeak\nresource\npressure",
    "Minimize\nworst\nrebuild\ncost",
    "Minimize\nspread\nbetween\nclasses",
]

fig, ax = plt.subplots(figsize=(10, 2.7))
ax.set_xlim(0, 10), ax.set_ylim(0, 2.7), ax.axis("off")

ax.add_patch(FancyBboxPatch((0.12, 0.12), 9.76, 2.46, boxstyle="round,pad=0,rounding_size=0.12",
                            fc=TEAL, ec="none"))
ax.text(0.42, 2.22, "Queue-Haul", color="white", style="italic", fontsize=22, fontweight="bold", va="center")

w, hh, yc, centers = 1.6, 0.62, 0.98, [1.55, 3.85, 6.15, 8.45]
for cx, label in zip(centers, STAGES):
    ax.add_patch(Rectangle((cx - w / 2, yc - hh), w, 2 * hh, fc="white", ec="black", lw=1.5))
    ax.text(cx, yc, label, ha="center", va="center", fontsize=13.5, color="black")
for cx in centers[:-1]:
    ax.add_patch(FancyArrowPatch((cx + w / 2, yc), (cx + 2.3 - w / 2, yc), arrowstyle="-|>",
                                 mutation_scale=22, lw=2.6, color="white"))

fig.savefig(OUT / "queue-haul-pipeline.pdf", bbox_inches="tight", pad_inches=0)
fig.savefig(OUT / "queue-haul-pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0)
print(f"wrote {OUT / 'queue-haul-pipeline.pdf'}")
