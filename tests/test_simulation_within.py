"""Error rates and interval coverage of withinstats' methods (SPEC.md section 11).

Each setting, its quantities, their kind (1: the nominal rate's 99% binomial band;
2: the band of the rate established by tools/reference/simulation_rates.py or
computed exactly) and the reason for the kind are in
``tests/simulation_settings.py`` (``WITHIN``); ``assert_rates`` prints every rate
with its band. The paired t and paired Hedges' g go through
``ip.analyze.withinstats``; the others through ``inferplot.stats``, which the
parity tests tie to the analysis function, because a bootstrap or a long
replication count through the public API would cost too much.
"""

from __future__ import annotations

import pytest

from tests import simulation_settings as ss

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("setting", ss.WITHIN, ids=lambda s: s.name)
def test_withinstats_rates(setting: ss.Setting) -> None:
    ss.assert_rates(setting)
