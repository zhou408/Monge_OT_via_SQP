"""Optional shared plotting helpers for notebook-derived experiments.

These utilities are intentionally lightweight and optional. The extracted
experiment scripts can keep their own plotting code unchanged.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_map_comparison(
    x: np.ndarray,
    learned: np.ndarray,
    analytic: Optional[np.ndarray] = None,
    title: str = "Transport Map",
    xlabel: str = "x",
    ylabel: str = "s(x)",
) -> None:
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(x, learned, linewidth=2, label="Learned map")
    if analytic is not None:
        plt.plot(x, analytic, "--", linewidth=2, label="Analytic map")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_density_comparison(
    y: np.ndarray,
    pushforward: np.ndarray,
    target: np.ndarray,
    title: str = "Target vs Pushforward Density",
    xlabel: str = "y",
    ylabel: str = "density",
) -> None:
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(y, target, "--", linewidth=2, label="Target density")
    plt.plot(y, pushforward, linewidth=2, label="Pushforward density")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

