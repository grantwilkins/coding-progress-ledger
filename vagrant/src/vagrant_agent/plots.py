"""Plot duplication factor across policies. Matplotlib defaults."""
from __future__ import annotations

from pathlib import Path


def plot_duplication_factor(summary: dict, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    policies = list(summary["policies"])
    factors = [summary["policies"][p]["cost_weighted_duplication_factor"] for p in policies]

    fig, ax = plt.subplots()
    ax.bar(policies, factors)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_ylabel("cost-weighted duplication factor")
    ax.set_xlabel(f"policy (tau={summary['tau']})")
    ax.set_title(
        f"shared-state duplication on toy trace\n"
        f"workflow {summary['manifest']['workflow_id']!r}, "
        f"{summary['manifest']['node_count']} nodes"
    )
    for i, v in enumerate(factors):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
