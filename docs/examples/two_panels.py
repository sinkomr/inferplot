"""Draw two analyses side by side in one matplotlib figure with ``ax=``.

Data: sleep from R's datasets package. Writes: two_panels.pdf
"""

import pandas as pd
from matplotlib.figure import Figure

import inferplot as ip

extra = {
    "A": [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0],
    "B": [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
}
df = pd.DataFrame(
    [(f"p{i + 1}", drug, h) for drug, hs in extra.items() for i, h in enumerate(hs)],
    columns=["patient", "drug", "extra"],
)
gain = pd.DataFrame(
    {"gain": [b - a for a, b in zip(extra["A"], extra["B"], strict=True)]}
)

# One panel above the other: the subtitles need the full width.
figure = Figure(figsize=(8, 11))
top, bottom = figure.subplots(2, 1, gridspec_kw={"hspace": 0.5})
paired = ip.withinstats(df, "drug", "extra", subject_id="patient", ax=top)
one = ip.histostats(gain, "gain", test_value=0, binwidth=0.5, ax=bottom)

# Each panel keeps its own result and checks; with ax= nothing is resized, so
# check() is how you learn that a text does not fit.
for fig in (paired, one):
    print(fig.result.text.apa)
    for issue in fig.check():
        print(issue)
# fig.save() writes the whole figure; it embeds the result of the panel it is
# called on.
paired.save("two_panels.pdf")
