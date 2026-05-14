#!/usr/bin/env python3
"""
Interactive HSV colour range sampler.

Click on a wire in the image window.  The script samples the HSV values in a
small patch around the click, widens them by a configurable tolerance, and
prints updated YAML ranges to stdout (or updates the config file in-place
with --write).

Uses matplotlib for the interactive window — avoids Qt/GTK backend issues
that affect some OpenCV builds.

Usage:
  python setup_colours.py panel.jpg
  python setup_colours.py panel.jpg --write --config config/colour_ranges.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Tolerance around the sampled mean (±H, ±S, ±V)
_TOLERANCE = (10, 50, 60)
_PATCH_RADIUS = 8  # pixels around click to sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample HSV ranges from a wire image.")
    p.add_argument("image", help="Panel photograph to sample from")
    p.add_argument("--config", default="config/colour_ranges.yaml")
    p.add_argument(
        "--write",
        action="store_true",
        help="Update the config file in-place with sampled ranges",
    )
    return p.parse_args()


def sample_patch(hsv: np.ndarray, cx: int, cy: int) -> dict:
    r = _PATCH_RADIUS
    h, w = hsv.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = hsv[y0:y1, x0:x1]

    h_vals = patch[:, :, 0].ravel().astype(float)
    s_vals = patch[:, :, 1].ravel().astype(float)
    v_vals = patch[:, :, 2].ravel().astype(float)

    th, ts, tv = _TOLERANCE
    return {
        "ranges": [[
            max(0,   int(h_vals.mean() - th)),
            max(0,   int(s_vals.mean() - ts)),
            max(0,   int(v_vals.mean() - tv)),
            min(180, int(h_vals.mean() + th)),
            min(255, int(s_vals.mean() + ts)),
            min(255, int(v_vals.mean() + tv)),
        ]]
    }


def main() -> int:
    args = parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        logger.error("Image not found: %s", img_path)
        return 1

    image = cv2.imread(str(img_path))
    if image is None:
        logger.error("Could not decode image: %s", img_path)
        return 1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as fh:
            config = yaml.safe_load(fh) or {}
    else:
        config = {"colours": {}}

    sampled: dict = {}

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.canvas.manager.set_window_title("Colour sampler — close window when done")
    ax.imshow(rgb)
    ax.set_title("Click on each wire, then type its colour name in the terminal.\nClose window when done.", fontsize=10)
    ax.axis("off")

    def on_click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        cx, cy = int(round(event.xdata)), int(round(event.ydata))

        colour_name = input(f"  Colour name for click ({cx}, {cy}): ").strip().lower()
        if not colour_name:
            return

        entry = sample_patch(hsv, cx, cy)
        sampled[colour_name] = entry
        r = entry["ranges"][0]
        logger.info(
            "Sampled '%s': H %d–%d  S %d–%d  V %d–%d",
            colour_name, r[0], r[3], r[1], r[4], r[2], r[5],
        )

        # Draw a marker and label on the plot
        ax.plot(cx, cy, "o", markersize=10, markeredgecolor="lime",
                markerfacecolor="none", markeredgewidth=2)
        ax.text(cx + _PATCH_RADIUS + 4, cy, colour_name,
                color="lime", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    print("Click on each wire colour then type its name in the terminal.")
    print("Close the image window when done.")
    plt.tight_layout()
    plt.show()   # blocks until the window is closed

    if not sampled:
        logger.info("No colours sampled.")
        return 0

    colours = config.setdefault("colours", {})
    for name, entry in sampled.items():
        colours[name] = entry

    yaml_out = yaml.dump({"colours": colours}, default_flow_style=False, sort_keys=False)

    if args.write:
        with open(config_path, "w") as fh:
            fh.write(yaml_out)
        logger.info("Config updated: %s", config_path)
    else:
        print("\n--- Updated colour_ranges.yaml ---")
        print(yaml_out)
        print("Run with --write to apply changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
