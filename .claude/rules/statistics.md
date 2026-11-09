---
paths:
  - "src/inferplot/stats/**"
  - "src/inferplot/analyses/**"
  - "tests/**"
  - "oracle/**"
---

# Statistics rules

- Before implementing a method, open its oracle fixture and the matching `SPEC.md` §4 row. Resolve every *(verify)* marker from the fixture first, then write the failing parity test, then the code.
- Each function in `stats/` cites its method in the docstring (author, year, equation or section) and names the R function it must match.
- `stats/` takes clean float64 NumPy arrays and returns frozen dataclasses of plain Python floats and ints. No dataframes, NaN handling, plotting or printing there.
- Compute upper-tail p-values with `sf`, never `1 - cdf`, also where R itself computes `1 - cdf` (WRS2's yuen, t1way and lincon): report the accurate tail, compare with the oracle under the tolerance that covers R's rounding (`wrs2_p`), and say so in the docstring (DECISIONS.md D43). Pass SciPy options by keyword.
- Randomness: `numpy.random.Generator(numpy.random.PCG64(seed))`, 2,000 resamples, the same CI construction R uses (percentile unless the fixture shows otherwise). Never reduce resamples to save time.
- Tolerances live only in `tests/tolerances.py` with a measured reason. Fixtures in `tests/fixtures/oracle/` are generated, never hand-edited.
- When a result disagrees with the oracle: check input parsing, then the method variant, then numerical precision. Record unresolved discrepancies in `docs/STATUS.md`; never paper over them.
- Mark tests over 5 seconds `@pytest.mark.slow`; simulation tests always are.
