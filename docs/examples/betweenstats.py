"""Compare independent groups: plant weight under a control and two treatments.

Data: PlantGrowth from R's datasets package. Writes: betweenstats.svg
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

# Independent groups, means compared: Welch's ANOVA with Games-Howell pairs.
fig = ip.betweenstats(df, "group", "weight", type="parametric", pairwise_display="all")
print(fig.result.text.apa)
print(fig.result)  # the APA sentence and the table of pairs
for issue in fig.check():
    print(issue)
for note in fig.result.warnings:
    print("note:", note)
fig.save("betweenstats.svg")
