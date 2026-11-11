"""Compare repeated measurements of the same subjects: two drugs, ten patients.

Data: sleep from R's datasets package (extra hours of sleep). Writes:
withinstats.svg
"""

import pandas as pd

import inferplot as ip

extra = {
    "A": [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0],
    "B": [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
}
# Long format: one row per patient and drug, with a column naming the patient.
df = pd.DataFrame(
    [(f"p{i + 1}", drug, h) for drug, hs in extra.items() for i, h in enumerate(hs)],
    columns=["patient", "drug", "extra"],
)

# Each patient took both drugs: pair the measurements by patient, not by row order.
fig = ip.withinstats(df, "drug", "extra", subject_id="patient", type="parametric")
print(fig.result.text.apa)
for issue in fig.check():
    print(issue)
for note in fig.result.warnings:
    print("note:", note)
fig.save("withinstats.svg")
