# Plotting rule

New or modified code that writes figures under `outputs/` must call
`plot_style.apply()` and use its canonical names, colors, and line styles for
registered concepts. Add a new shared identity to `plot_style.py`; do not
redefine it in a plot producer. Use local overrides only for plot-specific
layout or unregistered semantics.
