"""Ruler detection and pixels-per-mm calibration."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    pixels_per_mm: float
    tick_count: int
    ruler_line: Tuple[int, int, int, int]   # x1, y1, x2, y2 in full-image coords
    method: str                              # 'auto' or 'manual'


class CalibrationError(Exception):
    pass


class RulerDetector:
    """
    Detect a mm ruler in an image and return a pixels-per-mm scale factor.

    Auto-detection finds the longest horizontal line via Hough transform, then
    locates tick marks as periodic peaks in the perpendicular edge profile.
    Falls back to an interactive click-two-points window if auto fails.
    """

    def __init__(self, known_mm_per_tick: float = 1.0):
        self.known_mm_per_tick = known_mm_per_tick

    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> CalibrationResult:
        """
        Detect ruler and compute calibration.

        Parameters
        ----------
        image : full BGR image
        roi   : (x, y, w, h) sub-region containing the ruler; None = whole image
        """
        if roi:
            x, y, w, h = roi
            working = image[y : y + h, x : x + w]
        else:
            working = image

        try:
            result = self._auto_detect(working, roi)
            logger.info(
                "Auto calibration: %.3f px/mm from %d ticks",
                result.pixels_per_mm,
                result.tick_count,
            )
            return result
        except CalibrationError as exc:
            logger.warning("Auto calibration failed (%s) — switching to manual.", exc)
            return self._manual_detect(image)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auto_detect(
        self,
        roi_img: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> CalibrationResult:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Use 1/10 of width as minimum length — 1/4 is too strict for rulers that
        # don't span the full ROI or are interrupted by printed text/logos.
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=roi_img.shape[1] // 10,
            maxLineGap=50,
        )
        if lines is None:
            raise CalibrationError("No lines found")

        ruler_line = self._find_ruler_baseline(lines, roi_img.shape)
        spacings = self._find_tick_spacings(edges, ruler_line, roi_img.shape)

        if len(spacings) < 4:
            raise CalibrationError(
                f"Only {len(spacings) + 1} ticks detected (need ≥5)"
            )

        arr = np.array(spacings, dtype=float)
        # Remove outliers (>2× or <0.5× the median) before assessing regularity —
        # text and logos on the ruler body cause sporadic false peaks.
        med = np.median(arr)
        filtered = arr[(arr > med * 0.5) & (arr < med * 2.0)]
        if len(filtered) < 4:
            raise CalibrationError("Too few clean tick spacings after outlier removal")
        cv = filtered.std() / filtered.mean() if filtered.mean() > 0 else 1.0
        if cv > 0.45:
            raise CalibrationError(
                f"Tick spacing too irregular after filtering (CV={cv:.3f})"
            )

        px_per_mm = float(np.median(filtered) / self.known_mm_per_tick)

        # Translate line back to full-image coordinates if an ROI was used
        if roi:
            rx, ry = roi[0], roi[1]
            x1, y1, x2, y2 = ruler_line
            ruler_line = (x1 + rx, y1 + ry, x2 + rx, y2 + ry)

        return CalibrationResult(
            pixels_per_mm=px_per_mm,
            tick_count=int(len(filtered)) + 1,
            ruler_line=ruler_line,
            method="auto",
        )

    def _find_ruler_baseline(
        self, lines: np.ndarray, shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        w = shape[1]
        min_span = w * 0.20  # must span at least 20% of ROI width

        # Collect all near-horizontal lines meeting the minimum span requirement.
        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if not (angle < 5 or angle > 175):
                continue
            if float(np.hypot(x2 - x1, y2 - y1)) >= min_span:
                mid_y = (y1 + y2) / 2
                candidates.append((mid_y, x1, y1, x2, y2))

        if not candidates:
            raise CalibrationError("No horizontal line found for ruler baseline")

        # The ruler's tick edge is the topmost such line — tick marks protrude
        # upward from the ruler body, so the scale edge sits above the text/logo.
        candidates.sort(key=lambda c: c[0])
        _, x1, y1, x2, y2 = candidates[0]
        return (x1, y1, x2, y2)

    def _find_tick_spacings(
        self,
        edges: np.ndarray,
        ruler_line: Tuple[int, int, int, int],
        shape: Tuple[int, ...],
    ) -> list:
        x1, y1, x2, y2 = ruler_line
        h = shape[0]

        mid_y = (y1 + y2) // 2
        x_lo = min(x1, x2)
        x_hi = max(x1, x2)

        # Try sampling above the baseline first (ticks protrude upward), then
        # below (ticks protrude downward).  Use whichever side yields more
        # regular peaks — rulers placed at the bottom of frame often have ticks
        # below the detected top edge.
        tick_reach = 60

        def _profile_for_band(y_lo: int, y_hi: int) -> np.ndarray:
            b = edges[max(0, y_lo):min(h, y_hi), x_lo:x_hi]
            if b.size == 0:
                return np.zeros(x_hi - x_lo)
            p = b.sum(axis=0).astype(float)
            return uniform_filter1d(p, size=7)

        prof_above = _profile_for_band(mid_y - tick_reach, mid_y)
        prof_below = _profile_for_band(mid_y + 1, mid_y + tick_reach)

        # Score each side by peak count with high threshold (strong ticks only)
        def _peak_count(prof: np.ndarray) -> int:
            if prof.max() == 0:
                return 0
            pks, _ = find_peaks(prof, height=prof.max() * 0.5, distance=15)
            return len(pks)

        use_below = _peak_count(prof_below) > _peak_count(prof_above)
        profile = prof_below if use_below else prof_above
        band = edges[
            (mid_y + 1 if use_below else max(0, mid_y - tick_reach)) :
            (min(h, mid_y + tick_reach) if use_below else mid_y),
            x_lo:x_hi,
        ]
        if band.size == 0:
            raise CalibrationError("Ruler tick band is empty")

        profile = band.sum(axis=0).astype(float)
        profile = uniform_filter1d(profile, size=7)

        if profile.max() == 0:
            raise CalibrationError("No edges in ruler tick band")

        # Use a high threshold to catch only the tallest (longest) tick marks,
        # which are the most regularly spaced major graduations.
        peaks, _ = find_peaks(
            profile,
            height=profile.max() * 0.5,
            distance=15,
        )
        if len(peaks) < 2:
            raise CalibrationError(f"Fewer than 2 tick peaks detected (found {len(peaks)})")

        return np.diff(peaks).tolist()

    def _manual_detect(self, image: np.ndarray) -> CalibrationResult:
        """Interactive fallback: user clicks two points at a known distance."""
        points: list = []
        display = image.copy()

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                if len(points) == 2:
                    cv2.line(display, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Calibration", display)

        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(
            "Calibration",
            "Click two points at a known distance apart — press ESC to cancel",
        )
        cv2.imshow("Calibration", display)
        cv2.waitKey(1)   # pump event loop so window handle is valid before setMouseCallback
        cv2.setMouseCallback("Calibration", mouse_cb)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:
                cv2.destroyWindow("Calibration")
                raise CalibrationError("Manual calibration cancelled by user")
            if len(points) == 2:
                break

        cv2.destroyWindow("Calibration")

        dist_mm = float(input("Distance between the two clicked points in mm: "))
        dist_px = float(np.hypot(
            points[1][0] - points[0][0],
            points[1][1] - points[0][1],
        ))
        px_per_mm = dist_px / dist_mm

        x1, y1 = points[0]
        x2, y2 = points[1]
        return CalibrationResult(
            pixels_per_mm=px_per_mm,
            tick_count=2,
            ruler_line=(x1, y1, x2, y2),
            method="manual",
        )
