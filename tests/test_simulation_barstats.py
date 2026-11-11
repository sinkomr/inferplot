"""Error rates and interval coverage of barstats' methods (SPEC.md section 11).

Each setting, its quantities, their kind (1: the nominal rate's 99% binomial band;
2: the band of the rate computed exactly by enumeration or established by
tools/reference/simulation_rates.py) and the reason for the kind are in
``tests/simulation_settings.py`` (``BARSTATS``); ``assert_rates`` prints every
rate with its band. The 2 x 2 chi-square test, Cramer's V at no association and
check()'s small_expected go through ``ip.analyze.barstats``; the others through
``inferplot.stats`` on the table barstats tests (or its refusal), which
tests/test_simulation_barstats_settings.py ties to the analysis function table by
table.
"""

from __future__ import annotations

import pytest

from tests import simulation_settings as ss

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("setting", ss.BARSTATS, ids=lambda s: s.name)
def test_barstats_rates(setting: ss.Setting) -> None:
    ss.assert_rates(setting)
