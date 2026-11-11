"""Compare counts of categories: number of cylinders by transmission.

Data: mtcars from R's datasets package, given as a table of counts. Writes:
barstats.svg
"""

import pandas as pd

import inferplot as ip

table = pd.DataFrame(
    {
        "cyl": ["4", "4", "6", "6", "8", "8"],
        "am": ["automatic", "manual"] * 3,
        "cars": [3, 8, 4, 3, 12, 2],
    }
)

# One bar per transmission, filled by the shares of cylinder counts. Each row of
# the table stands for `cars` cars; with one row per car, leave counts= out.
fig = ip.barstats(table, "cyl", "am", counts="cars")
print(fig.result.text.apa)
for issue in fig.check():
    print(issue)
for note in fig.result.warnings:
    print("note:", note)
fig.save("barstats.svg")
