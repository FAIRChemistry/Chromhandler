"""Handler-level chromatogram plotting.

Provides two figure builders:

- :func:`plot_traces`: one signal panel per group (overlay = single / sample / all).
- :func:`plot_window_grid`: rows x windows grid with the same overlay semantics.

Plus small internal helpers (:func:`_group_chromatograms`, :func:`_line_colors`)
that drive both builders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from chromhandler.handler import Handler
    from chromhandler.model import Chromatogram

OverlayMode = Literal["all", "sample", "single"]


def _group_chromatograms(handler: Handler, overlay: OverlayMode) -> list[list[Chromatogram]]:
    """Flatten ``handler.samples`` → chromatograms into grouped rows.

    - ``"single"``: one group per chromatogram (flat).
    - ``"sample"``: one group per sample, in handler order.
    - ``"all"``: one group containing every chromatogram (flat).

    Raises:
        ValueError: If the handler has no chromatograms across any sample.
    """
    flat: list[Chromatogram] = []
    per_sample: list[list[Chromatogram]] = []
    for sample in handler.samples:
        if not sample.chromatograms:
            continue
        per_sample.append(list(sample.chromatograms))
        flat.extend(sample.chromatograms)
    if not flat:
        raise ValueError("Handler has no chromatograms across any sample.")
    if overlay == "single":
        return [[c] for c in flat]
    if overlay == "sample":
        return per_sample
    if overlay == "all":
        return [flat]
    raise ValueError(f"Unknown overlay mode: {overlay!r}")


def _line_colors(n: int) -> list[tuple[float, float, float, float]]:
    """Return ``n`` line colors per the project rule.

    1 line → ``tab:blue``; ≥ 2 lines → viridis evenly spaced over ``[0, 1]``.
    """
    if n <= 0:
        return []
    if n == 1:
        return [mcolors.to_rgba("tab:blue")]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / (n - 1)) for i in range(n)]
