#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib", "tyro", "rasterio"]
# ///
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import tyro
from matplotlib.patches import Rectangle


def plot_tiles(json_file: Path, mask: Path | None = None) -> None:
    """
    Plot tile boxes from a JSON file over a mask image.

    The JSON file is expected to contain a dictionary with a "shape" key
    (a two-element list or tuple) and a "tiles" key. Each tile is a dictionary
    with a "bounds" key, where bounds are specified as a four-element list:
    [top, left, bottom, right]. The function reads the mask image from the given
    path and displays it as the background, then overlays rectangles corresponding
    to the tile bounds.

    Parameters
    ----------
    json_file : Path
        Path to the JSON file containing the tile data.
    mask : Path, optional
        Path to the mask image file to be used as the background.

    Returns
    -------
    None

    Examples
    --------
    >>> from pathlib import Path
    >>> plot_tiles(Path("tiles.json"), Path("mask.png"))
    """
    # Load tile data from JSON file
    with json_file.open(mode="r") as fid:
        data = json.load(fid)

    shape = data["shape"]

    # Create figure and axis with fixed limits based on the shape in the JSON data.
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_ylim(0, shape[0])
    ax.set_xlim(0, shape[1])

    # Plot the mask image first.
    # The extent parameter ensures the mask covers the full area.
    if mask:
        with rio.open(mask) as src:
            mask_image = np.ma.masked_equal(src.read(1), 0)
        # ax.imshow(mask_image, extent=[0, shape[1], 0, shape[0]], origin="upper")
        ax.imshow(mask_image, cmap="gray")

    # Plot tile boxes from the JSON data.
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(data["tiles"])))
    for tile, cur_color in zip(data["tiles"], colors):
        p = tile["bounds"]
        rect = Rectangle(
            (
                p[1],
                p[0],
            ),  # (x, y) lower-left corner; note that x corresponds to the left bound.
            p[3] - p[1],  # width: right bound - left bound
            p[2] - p[0],  # height: bottom bound - top bound
            facecolor="none",
            linewidth=2,
            linestyle="dashed",
            edgecolor=cur_color,
        )
        ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    tyro.cli(plot_tiles)
