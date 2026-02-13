"""Data classes for chromatography fitting."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class PeakWindow:
    x1: float  # x_min of the window
    x2: float  # x_max of the window


@dataclass
class Peak:
    """
    Stores params of a exponential modified gaussian peak
    """

    id: str
    A: float
    mu: float
    sigma: float
    tau: float


@dataclass
class ChromatogramSection:
    id: str
    x_arr: jnp.ndarray
    y_arr: jnp.ndarray
    windows: list[PeakWindow]


@dataclass
class Chromatogram:
    id: str
    x_arr: jnp.ndarray
    y_arr: jnp.ndarray
    sections: list[ChromatogramSection] = []
    peaks: list[Peak] = []
