---
paths:
  - "src/inferplot/render/**"
  - "src/inferplot/export.py"
  - "tests/render/**"
---

# Rendering rules

- matplotlib object-oriented API only: `matplotlib.figure.Figure` with `FigureCanvasAgg`. Never import `pyplot`, call `show()`, or touch global `rcParams`; use `rc_context` locally.
- Every number drawn comes from the `StatsResult`; renderers compute no statistics.
- Text: mathtext only (no `usetex`); default font DejaVu Sans, which ships with matplotlib, so sandboxes render identically.
- Semantic parity with ggstatsplot: same layers and the same subtitle and caption content; the look is ours (Okabe-Ito palette, clean axes, n per group in x tick labels).
- Significance brackets: own algorithm. Shortest spans lowest; levels spaced in points (label height, padding, legs and line width) on top of the data region (D38); label above each bracket; extend the y-limit to fit. No overlaps.
- Jitter is seeded from `seed`. More than 5,000 points in a panel: density layer plus a seeded 2,000-point subsample, recorded in `warnings`.
- `check()` measures overlap and clipping from rendered text bounding boxes after `canvas.draw()`.
- Tests: structural SVG assertions (layers present, labels equal result text) for every function on every OS; pytest-mpl image comparisons only for a few cases, run and baselined in the Linux CI job only.
