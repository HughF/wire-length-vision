"""HSV colour segmentation — extracts per-colour wire masks from a panel image."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger(__name__)


@dataclass
class WireInstance:
    colour: str
    wire_id: str
    mask: np.ndarray                        # uint8 binary mask (0 / 255)
    area_px: int
    bbox: Tuple[int, int, int, int]         # x, y, w, h
    status_flags: List[str] = field(default_factory=list)


class WireSegmenter:
    """
    Segment wires by colour using per-colour HSV range definitions loaded from
    a YAML config file.  Each disconnected blob of a given colour is returned
    as a separate WireInstance.
    """

    def __init__(self, config_path: str, min_area_px: int = 500):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Colour config not found: {config_path}")
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        self._colour_cfg: Dict = cfg.get("colours", {})
        self.min_area_px = min_area_px

    def segment(
        self, image: np.ndarray, apply_clahe: bool = True
    ) -> List[WireInstance]:
        """
        Segment all configured wire colours in *image*.

        Returns a flat list of WireInstance objects, one per connected blob.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        colour_masks: Dict[str, np.ndarray] = {}
        for name, defn in self._colour_cfg.items():
            mask = self._build_mask(hsv, defn)
            if mask is not None and mask.any():
                colour_masks[name] = mask

        self._warn_conflicts(colour_masks)

        wires: List[WireInstance] = []
        for name, mask in colour_masks.items():
            wires.extend(self._extract_instances(mask, name, hsv))

        if not wires:
            logger.warning(
                "No wire instances found. "
                "Check colour_ranges.yaml and image lighting."
            )
        return wires

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_mask(
        self, hsv: np.ndarray, defn: dict
    ) -> Optional[np.ndarray]:
        ranges = defn.get("ranges", [])
        if not ranges:
            return None

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lo = np.array(r[:3], dtype=np.uint8)
            hi = np.array(r[3:], dtype=np.uint8)
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

        # Morphological close: bridge specular highlight gaps inside the wire
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)

        # Fill internal holes caused by reflections
        filled = binary_fill_holes(combined.astype(bool))
        return filled.astype(np.uint8) * 255

    def _warn_conflicts(self, masks: Dict[str, np.ndarray]) -> None:
        names = list(masks.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = masks[names[i]], masks[names[j]]
                overlap = int(cv2.bitwise_and(a, b).astype(bool).sum())
                smaller = min(int(a.astype(bool).sum()), int(b.astype(bool).sum()))
                if smaller > 0 and overlap / smaller > 0.02:
                    logger.warning(
                        "ColourConflict: '%s' and '%s' masks overlap by %.1f%% — "
                        "tighten HSV ranges in colour_ranges.yaml",
                        names[i],
                        names[j],
                        overlap / smaller * 100,
                    )

    def _extract_instances(
        self,
        mask: np.ndarray,
        colour: str,
        hsv: np.ndarray,
    ) -> List[WireInstance]:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        instances: List[WireInstance] = []
        wire_idx = 1

        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue

            instance_mask = (labels == label_id).astype(np.uint8) * 255
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            flags: List[str] = []
            v_vals = hsv[:, :, 2][instance_mask > 0]
            if v_vals.size > 0 and float(v_vals.mean()) > 240:
                flags.append("high_specular")

            instances.append(
                WireInstance(
                    colour=colour,
                    wire_id=f"{colour}_{wire_idx}",
                    mask=instance_mask,
                    area_px=area,
                    bbox=(x, y, w, h),
                    status_flags=flags,
                )
            )
            wire_idx += 1

        return instances
