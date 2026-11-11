"""``fig.show()`` says how to look at a figure (Phase 5 review, U1)."""

from __future__ import annotations

import pandas as pd
import pytest

import inferplot as ip


def test_show_raises_with_the_way_to_look_at_the_figure() -> None:
    frame = pd.DataFrame({"y": [1.0, 2.0, 2.5, 3.1, 4.0, 2.2]})
    fig = ip.histostats(frame, "y")
    with pytest.raises(RuntimeError) as caught:
        fig.show()
    message = str(caught.value)
    assert "fig.save('figure.png')" in message
    assert "notebook" in message
    # What the message advises works.
    assert fig._repr_png_().startswith(b"\x89PNG")
