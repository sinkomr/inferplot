"""Compute once with ``ip.analyze``, draw twice with ``ip.render``.

Data: PlantGrowth from R's datasets package. Writes: weight_default.svg,
weight_boxes.svg
"""

import pandas as pd

import inferplot as ip

weights = {
    "ctrl": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14],
    "trt1": [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69],
    "trt2": [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26],
}
df = pd.DataFrame(
    [(group, w) for group, ws in weights.items() for w in ws],
    columns=["group", "weight"],
)

# The analysis imports no matplotlib: every number and text is in the result.
result = ip.analyze.betweenstats(df, "group", "weight", type="nonparametric")
print(result.text.apa)
print(result.to_frame()[["test", "group1", "group2", "p_adjusted"]])

# Drawing options belong to the render call; the numbers do not change.
plain = ip.render.betweenstats(result)
boxes = ip.render.betweenstats(
    result, layers=("box", "points"), palette="viridis", title="Plant growth"
)
for fig, path in ((plain, "weight_default.svg"), (boxes, "weight_boxes.svg")):
    for issue in fig.check():
        print(issue)
    fig.save(path)
