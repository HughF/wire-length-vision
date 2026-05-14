#!/usr/bin/env python3
"""Wire Length Vision — entry point.

Usage examples:
  python main.py panel.jpg
  python main.py panel.jpg --mm-per-tick 10 --roi 0 800 1200 80
  python main.py panel.jpg --precise --output results/
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from wire_length_vision.calibration.ruler_detector import CalibrationError, CalibrationResult, RulerDetector
from wire_length_vision.measurement.skeleton_tracer import SkeletonTracer
from wire_length_vision.output.reporter import Reporter
from wire_length_vision.segmentation.wire_segmenter import WireSegmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure wire lengths from a panel photograph."
    )
    p.add_argument("image", help="Path to the panel photograph (JPEG or PNG)")
    p.add_argument(
        "--config",
        default="config/colour_ranges.yaml",
        help="HSV colour config YAML  (default: config/colour_ranges.yaml)",
    )
    p.add_argument(
        "--output",
        default="output",
        help="Output directory for results  (default: ./output)",
    )
    p.add_argument(
        "--mm-per-tick",
        type=float,
        default=1.0,
        metavar="MM",
        help="Ruler tick spacing in mm  (default: 1.0)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=500,
        metavar="PX",
        help="Minimum wire mask area in pixels  (default: 500)",
    )
    p.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Ruler region-of-interest in pixels  e.g. --roi 0 820 1200 60",
    )
    p.add_argument(
        "--precise",
        action="store_true",
        help="Use medial_axis skeletonisation (slower, more accurate on thick wires)",
    )
    p.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE brightness normalisation",
    )
    p.add_argument(
        "--px-per-mm",
        type=float,
        default=None,
        metavar="SCALE",
        help="Skip ruler auto-detection and use this fixed scale (pixels per mm)",
    )
    return p.parse_args()


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

    logger.info("Image: %s  (%d × %d px)", img_path.name, image.shape[1], image.shape[0])

    # ------------------------------------------------------------------
    # Step 1 — calibrate from ruler (or use fixed scale if provided)
    # ------------------------------------------------------------------
    if args.px_per_mm is not None:
        h, w = image.shape[:2]
        calibration = CalibrationResult(
            pixels_per_mm=args.px_per_mm,
            tick_count=0,
            ruler_line=(0, h - 1, w - 1, h - 1),
            method="manual",
        )
        logger.info("Using fixed scale: %.3f px/mm", calibration.pixels_per_mm)
    else:
        roi = tuple(args.roi) if args.roi else None
        detector = RulerDetector(known_mm_per_tick=args.mm_per_tick)
        try:
            calibration = detector.detect(image, roi=roi)
        except CalibrationError as exc:
            logger.error("Calibration failed: %s", exc)
            return 1
        logger.info(
            "Calibration: %.3f px/mm  (%s, %d ticks)",
            calibration.pixels_per_mm,
            calibration.method,
            calibration.tick_count,
        )

    # ------------------------------------------------------------------
    # Step 2 — segment wires by colour
    # ------------------------------------------------------------------
    segmenter = WireSegmenter(args.config, min_area_px=args.min_area)
    wires = segmenter.segment(image, apply_clahe=not args.no_clahe)
    logger.info("Detected %d wire instance(s)", len(wires))

    if not wires:
        logger.warning("No wires detected — check colour_ranges.yaml")
        return 0

    # ------------------------------------------------------------------
    # Step 3 — measure each wire
    # ------------------------------------------------------------------
    tracer = SkeletonTracer(
        pixels_per_mm=calibration.pixels_per_mm,
        precise=args.precise,
    )
    measurements = [tracer.measure(w) for w in wires]

    # ------------------------------------------------------------------
    # Step 4 — save outputs
    # ------------------------------------------------------------------
    reporter = Reporter(args.output)
    img_out, json_out, csv_out = reporter.save_all(
        image, measurements, calibration, str(img_path)
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print(f"{'Wire':<14} {'Length':>10}   Flags")
    print("-" * 45)
    for m in sorted(measurements, key=lambda x: x.wire_id):
        flags_str = ", ".join(m.status_flags) if m.status_flags else "ok"
        print(f"{m.wire_id:<14} {m.length_mm:>9.1f} mm   {flags_str}")
    print()
    print(f"Outputs written to: {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
