"""Module for controlling visualization-related processing."""

import os

import matplotlib.pyplot as plt
import seaborn as sns  # pyright: ignore

__all__ = ["init_visualization"]


def init_visualization(font: str | None = None) -> None:
    """Initialize visualization libraries (matplotlib, seaborn).

    Args:
        font (str, optional): Font name
            (Default: "Meiryo" for Windows, "Noto Sans CJK JP" otherwise)
    """
    if not font:
        font = "Meiryo" if os.name == "nt" else "Noto Sans CJK JP"

    plt.rcParams["axes.formatter.useoffset"] = False
    plt.rcParams["font.family"] = font
    sns.set(font=font)  # pyright: ignore
