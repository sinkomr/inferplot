"""Test one sample against a value: is the cars' mean fuel use 20 miles per gallon?

Data: mtcars from R's datasets package. Writes: histostats.svg
"""

import pandas as pd

import inferplot as ip

mpg = [
    21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8,
    16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5,
    15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4,
]  # fmt: skip
df = pd.DataFrame({"mpg": mpg})

# One sample against a fixed value: the one-sample t-test with Hedges' g.
fig = ip.histostats(df, "mpg", test_value=20, type="parametric")
print(fig.result.text.apa)
for issue in fig.check():
    print(issue)
for note in fig.result.warnings:
    print("note:", note)
fig.save("histostats.svg")
