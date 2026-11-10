"""Error rates and interval coverage of histostats' methods (SPEC.md section 11).

Each setting, its quantities, their kind (1: the nominal rate's 99% binomial band;
2: the band of the rate established by tools/reference/simulation_rates.py or
computed exactly) and the reason for the kind are in
``tests/simulation_settings.py`` (``HISTOSTATS``); ``assert_rates`` prints every
rate with its band. The two-sided one-sample t and Hedges' g at no effect go
through ``ip.analyze.histostats``; the others through ``inferplot.stats``.
"""

from __future__ import annotations

import pytest

from tests import simulation_settings as ss

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("setting", ss.HISTOSTATS, ids=lambda s: s.name)
def test_histostats_rates(setting: ss.Setting) -> None:
    ss.assert_rates(setting)
