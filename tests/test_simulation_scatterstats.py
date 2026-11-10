"""Error rates and interval coverage of scatterstats' methods (SPEC.md section 11).

Each setting, its quantities, their kind (1: the nominal rate's 99% binomial band;
2: the band of the rate established by tools/reference/simulation_rates.py) and
the reason for the kind are in ``tests/simulation_settings.py``
(``SCATTERSTATS``), with the derivation of the true Spearman's rho and 20%
winsorized correlation of a bivariate normal; ``assert_rates`` prints every rate
with its band. Pearson's test and interval at rho 0 go through
``ip.analyze.scatterstats``; the others through ``inferplot.stats``.
"""

from __future__ import annotations

import pytest

from tests import simulation_settings as ss

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("setting", ss.SCATTERSTATS, ids=lambda s: s.name)
def test_scatterstats_rates(setting: ss.Setting) -> None:
    ss.assert_rates(setting)
