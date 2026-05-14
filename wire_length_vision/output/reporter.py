"""Generate annotated image overlays and CSV/JSON result files."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR draw colours for each wire colour name
_OVERLAY_BGR = {
    "red":    (30,  30,  220),
    "blue":   (220, 80,  0),
    "green":  (20,  180, 20),
    "yellow": (0,   200, 220),
    "orange": (0,   128, 255),
    "white":  (200, 200, 200),
    "black":  (80,  80,  80),
    "brown":  (30,  60,  100),
    "grey":   (150, 150, 150),
    "purple": (180, 0,   180),
}
_FALLBACK_BGR = (255, 0, 255)


def _bgr(colour: str) -> Tuple[int, int, int]:
    return _OVERLAY_BGR.get(colour, _FALLBACK_BGR)


class Reporter:
    """Write annotated PNG, JSON, and CSV outputs to *output_dir*."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_all(
        self, image: np.ndarray, measurements, calibration, image_path: str
    ) -> Tuple[Path, Path, Path]:
        stem = Path(image_path).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stem}_{ts}"

        annotated = self._draw_overlay(image, measurements, calibration)

        img_out = self.output_dir / f"{base}_annotated.png"
        json_out = self.output_dir / f"{base}_results.json"
        csv_out = self.output_dir / f"{base}_results.csv"

        cv2.imwrite(str(img_out), annotated)
        self._save_json(measurements, calibration, image_path, json_out)
        self._save_csv(measurements, csv_out)

        logger.info("Annotated image : %s", img_out)
        logger.info("JSON results    : %s", json_out)
        logger.info("CSV results     : %s", csv_out)

        return img_out, json_out, csv_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_overlay(self, image: np.ndarray, measurements, calibration) -> np.ndarray:
        out = image.copy()

        for m in measurements:
            if not m.skeleton_path:
                continue
            colour_bgr = _bgr(m.colour)
            # skeleton_path stores (row, col); convert to (x, y) for OpenCV
            pts = [(col, row) for (row, col) in m.skeleton_path]

            for i in range(len(pts) - 1):
                cv2.line(out, pts[i], pts[i + 1], colour_bgr, 2, cv2.LINE_AA)

            mid = pts[len(pts) // 2]
            label = f"{m.wire_id}: {m.length_mm:.0f} mm"
            cv2.putText(
                out, label,
                (mid[0] + 6, mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, colour_bgr, 2, cv2.LINE_AA,
            )

            if "crossing_interpolated" in m.status_flags:
                cv2.circle(out, mid, 12, (0, 220, 220), 2)
                cv2.putText(
                    out, "!",
                    (mid[0] - 4, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 220), 2, cv2.LINE_AA,
                )

        # Draw ruler annotation
        x1, y1, x2, y2 = calibration.ruler_line
        cv2.line(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cal_label = (
            f"Ruler: {calibration.pixels_per_mm:.2f} px/mm"
            f"  ({calibration.method})"
        )
        cv2.putText(
            out, cal_label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 220, 0), 2, cv2.LINE_AA,
        )

        return out

    def _save_json(self, measurements, calibration, image_path: str, out_path: Path):
        data = {
            "image": image_path,
            "timestamp": datetime.now().isoformat(),
            "calibration": {
                "pixels_per_mm": calibration.pixels_per_mm,
                "tick_count": calibration.tick_count,
                "method": calibration.method,
            },
            "wires": [
                {
                    "wire_id": m.wire_id,
                    "colour": m.colour,
                    "length_mm": round(m.length_mm, 2),
                    "length_px": round(m.length_px, 2),
                    "flags": m.status_flags,
                }
                for m in measurements
            ],
        }
        with open(out_path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _save_csv(self, measurements, out_path: Path):
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["colour", "wire_id", "length_mm", "flags"])
            for m in measurements:
                writer.writerow([
                    m.colour,
                    m.wire_id,
                    f"{m.length_mm:.2f}",
                    "|".join(m.status_flags) if m.status_flags else "ok",
                ])
