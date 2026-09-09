"""Render frozen results with complete percentage axes; preserve execution source hashes."""

from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure

import plot_style
import pool_shed_cache_sensitivity as campaign

savefig = Figure.savefig


def save(figure, path, **kwargs):
    if Path(path).stem in ("handoff_fraction", "kv_source_fraction"):
        for axes in figure.axes:
            axes.set_ylim(0, 105)
    return savefig(figure, path, **kwargs)


plot_style.apply()
with patch.object(Figure, "savefig", save):
    campaign.reduce(Path(__file__).resolve().parent)
