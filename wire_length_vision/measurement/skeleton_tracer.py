"""Skeletonise wire masks and compute arc-length measurements."""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import networkx as nx
import numpy as np
from skimage.morphology import medial_axis, skeletonize

logger = logging.getLogger(__name__)


@dataclass
class WireMeasurement:
    wire_id: str
    colour: str
    length_px: float
    length_mm: float
    status_flags: List[str] = field(default_factory=list)
    # Representative centreline for drawing the overlay
    skeleton_path: List[Tuple[int, int]] = field(default_factory=list)


class SkeletonTracer:
    """
    Convert a binary wire mask into a skeleton graph, prune artefact spurs,
    and sum all edge lengths to get the total wire arc-length.
    """

    def __init__(
        self,
        pixels_per_mm: float,
        spur_threshold_px: int = 5,
        precise: bool = False,
    ):
        self.pixels_per_mm = pixels_per_mm
        self.spur_threshold_px = spur_threshold_px
        self.precise = precise

    def measure(self, wire) -> WireMeasurement:
        """Measure a single WireInstance and return a WireMeasurement."""
        mask_bool = wire.mask.astype(bool)

        if self.precise:
            skel, _ = medial_axis(mask_bool, return_distance=True)
        else:
            skel = skeletonize(mask_bool)

        if not skel.any():
            logger.warning("%s: empty skeleton", wire.wire_id)
            return WireMeasurement(
                wire_id=wire.wire_id,
                colour=wire.colour,
                length_px=0.0,
                length_mm=0.0,
                status_flags=wire.status_flags + ["empty_skeleton"],
            )

        G = self._build_graph(skel)
        G = self._prune_spurs(G)

        total_px = sum(d["length"] for _, _, d in G.edges(data=True))
        path = self._representative_path(G)

        flags = list(wire.status_flags)
        junction_nodes = [n for n, deg in G.degree() if deg >= 3]
        if junction_nodes:
            flags.append("crossing_interpolated")

        return WireMeasurement(
            wire_id=wire.wire_id,
            colour=wire.colour,
            length_px=total_px,
            length_mm=total_px / self.pixels_per_mm,
            status_flags=flags,
            skeleton_path=path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_graph(self, skel: np.ndarray) -> nx.Graph:
        """Build a pixel-level graph from a boolean skeleton array."""
        G: nx.Graph = nx.Graph()
        ys, xs = np.where(skel)
        px_set = set(zip(map(int, ys), map(int, xs)))

        G.add_nodes_from(px_set)

        for y, x in px_set:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb = (y + dy, x + dx)
                    if nb in px_set and not G.has_edge((y, x), nb):
                        G.add_edge((y, x), nb, length=float(np.hypot(dy, dx)))

        return G

    def _prune_spurs(self, G: nx.Graph) -> nx.Graph:
        """
        Iteratively remove short dead-end branches (skeleton artefacts at
        junctions).  Stops when no more spurs are shorter than the threshold.
        """
        changed = True
        while changed:
            changed = False
            leaves = [n for n in list(G.nodes()) if G.degree(n) == 1]
            for leaf in leaves:
                if not G.has_node(leaf):
                    continue
                path = [leaf]
                cur = leaf
                while True:
                    neighbours = [
                        nb for nb in G.neighbors(cur)
                        if nb != (path[-2] if len(path) > 1 else None)
                    ]
                    if len(neighbours) != 1:
                        break
                    nxt = neighbours[0]
                    if G.degree(nxt) >= 3:
                        break
                    path.append(nxt)
                    cur = nxt

                spur_len = sum(
                    G[path[i]][path[i + 1]]["length"]
                    for i in range(len(path) - 1)
                )
                if spur_len < self.spur_threshold_px:
                    for node in path:
                        if G.has_node(node):
                            G.remove_node(node)
                    changed = True
                    break   # restart the leaf scan after mutation

        return G

    def _representative_path(self, G: nx.Graph) -> List[Tuple[int, int]]:
        """Return the longest simple path between any two endpoints for overlay rendering."""
        if G.number_of_nodes() == 0:
            return []

        endpoints = [n for n in G.nodes() if G.degree(n) == 1]

        # Fall back to any two nodes if no clear endpoints
        if len(endpoints) < 2:
            nodes = list(G.nodes())
            endpoints = nodes[:2] if len(nodes) >= 2 else nodes
        if len(endpoints) < 2:
            return list(G.nodes())

        # Try shortest path between each pair of endpoints; keep the longest
        best: List = []
        for i in range(min(len(endpoints), 8)):
            for j in range(i + 1, min(len(endpoints), 8)):
                try:
                    p = nx.shortest_path(G, endpoints[i], endpoints[j])
                    if len(p) > len(best):
                        best = p
                except nx.NetworkXNoPath:
                    continue

        return best if best else list(G.nodes())[:200]
