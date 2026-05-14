"""Smoke tests using synthetic images — no real panel photograph required."""

import numpy as np
import pytest

from wire_length_vision.measurement.skeleton_tracer import SkeletonTracer, WireMeasurement
from wire_length_vision.segmentation.wire_segmenter import WireInstance, WireSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return m


def _wire_instance(colour: str = "red", h: int = 200, w: int = 600) -> WireInstance:
    """Horizontal bar mask — should produce a skeleton ~w/2 pixels long."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - 5 : h // 2 + 5, 50 : w - 50] = 255
    return WireInstance(
        colour=colour,
        wire_id=f"{colour}_1",
        mask=mask,
        area_px=int(mask.astype(bool).sum()),
        bbox=(50, h // 2 - 5, w - 100, 10),
    )


# ---------------------------------------------------------------------------
# SkeletonTracer
# ---------------------------------------------------------------------------

class TestSkeletonTracer:
    def test_horizontal_wire_length_approximate(self):
        wire = _wire_instance(w=500)
        tracer = SkeletonTracer(pixels_per_mm=10.0)
        m = tracer.measure(wire)
        # Expect roughly (500 - 100) px long skeleton → 40 mm ± 10%
        assert 36.0 <= m.length_mm <= 44.0, f"Unexpected length: {m.length_mm:.1f} mm"

    def test_empty_mask_returns_zero(self):
        wire = WireInstance(
            colour="blue",
            wire_id="blue_1",
            mask=np.zeros((100, 100), dtype=np.uint8),
            area_px=0,
            bbox=(0, 0, 100, 100),
        )
        tracer = SkeletonTracer(pixels_per_mm=5.0)
        m = tracer.measure(wire)
        assert m.length_mm == 0.0
        assert "empty_skeleton" in m.status_flags

    def test_pixels_per_mm_scaling(self):
        wire = _wire_instance(w=500)
        m1 = SkeletonTracer(pixels_per_mm=10.0).measure(wire)
        m2 = SkeletonTracer(pixels_per_mm=20.0).measure(wire)
        assert abs(m1.length_mm - 2 * m2.length_mm) < 1.0


# ---------------------------------------------------------------------------
# WireSegmenter (requires a synthetic BGR image matching our YAML ranges)
# ---------------------------------------------------------------------------

class TestWireSegmenter:
    CONFIG = "config/colour_ranges.yaml"

    def _make_panel(self, colour_bgr: tuple, line_y: int = 100) -> np.ndarray:
        """Solid grey panel with a single horizontal coloured wire stripe."""
        img = np.full((200, 400, 3), 80, dtype=np.uint8)
        img[line_y - 5 : line_y + 5, 20:380] = colour_bgr
        return img

    def test_detects_blue_wire(self):
        # Pure blue BGR=(255,0,0) → HSV H≈120
        img = self._make_panel((255, 0, 0))
        seg = WireSegmenter(self.CONFIG, min_area_px=100)
        wires = seg.segment(img, apply_clahe=False)
        colours = {w.colour for w in wires}
        assert "blue" in colours, f"Expected blue, got {colours}"

    def test_detects_green_wire(self):
        # Pure green BGR=(0,255,0) → HSV H≈60
        img = self._make_panel((0, 255, 0))
        seg = WireSegmenter(self.CONFIG, min_area_px=100)
        wires = seg.segment(img, apply_clahe=False)
        colours = {w.colour for w in wires}
        assert "green" in colours, f"Expected green, got {colours}"

    def test_min_area_filter(self):
        img = self._make_panel((255, 0, 0))
        seg = WireSegmenter(self.CONFIG, min_area_px=99999)
        wires = seg.segment(img, apply_clahe=False)
        assert wires == [], "Expected no wires above huge area threshold"
